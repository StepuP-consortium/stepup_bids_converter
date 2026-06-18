"""Shared XDF -> BIDS conversion engine for the StepUp sites.

This module holds the site-agnostic conversion logic; the per-site specifics
(file discovery, task routing, montage cap, mocap clusterer, EMG map) are
supplied through a :class:`SiteConfig` by the thin site modules
(:mod:`src.bids_kiel`, :mod:`src.bids_telaviv`).

Pipeline per file (see :func:`run_conversion`):
    EEG  -> always (defines the common LSL time window)
    EMG  -> walking tasks only, cropped to the EEG window
    Mocap-> walking tasks only, cropped to the EEG window, clustered to markers
Each modality is isolated so one failure never blocks the others, and a compact
batch summary lists everything that needs attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")  # non-blocking: diagnostic plots are saved, not shown
import matplotlib.pyplot as plt
import mne
import numpy as np
import pyxdf
from mne_bids import make_dataset_description, write_raw_bids, update_sidecar_json, BIDSPath
from mnelab.io.xdf import read_raw_xdf

from motionbids import (
    MotionData,
    Channel,
    export_bids_motion,
    validate_motion_data,
    create_bids_directory_structure,
)

from .xdf import get_effective_srate_xdf


# --------------------------------------------------------------------------- #
# Per-site configuration
# --------------------------------------------------------------------------- #
@dataclass
class SiteConfig:
    """Everything the engine needs that differs between sites."""

    bids_root: Path
    dataset_name: str
    qc_dir: Path
    line_freq: int = 50

    # EEG montage
    bvef_path: Optional[Path] = None        # 129-ch cap file (Kiel); None elsewhere
    eeg_rename_labels: Optional[list] = None  # candidate ordered names (Tel Aviv g.tec)
    apply_eeg_rename: bool = False          # apply eeg_rename_labels + standard_1005

    # EMG
    emg_map: dict = field(default_factory=dict)  # stream channel -> muscle name
    emg_scale: float = 1.0                       # multiply raw EMG (e.g. 1e-3 for mV->V)

    # Mocap
    mocap_stream_type: str = "6D"           # "6D" (Qualisys) or "MoCap" (Vicon)
    clusterer: Optional[Callable] = None    # cluster_markers_tracked / _vicon
    cluster_kwargs: dict = field(default_factory=dict)
    tracksys: str = "qualisys"
    manufacturer: str = "Qualisys"


# --------------------------------------------------------------------------- #
# Stream / montage helpers
# --------------------------------------------------------------------------- #
def stream_by_type(streams, stype):
    """Return the first stream of the given LSL type, or None."""
    matches = [s for s in streams if s["info"]["type"][0] == stype]
    return matches[0] if matches else None


def parse_recording_datetime(header):
    """Parse the XDF header recording-start datetime as a UTC-aware datetime.

    Returns ``None`` if the header has no usable datetime (then acq_time stays
    n/a). This is the wall-clock anchor for the EEG stream; the EMG/mocap files
    get this time plus their LSL offset relative to EEG (see :func:`run_conversion`).
    """
    try:
        s = header["info"]["datetime"][0]
    except (KeyError, IndexError, TypeError):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(s, fmt).astimezone(timezone.utc)
        except (ValueError, TypeError):
            pass
    try:
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _acq_time_str(dt):
    """Format a datetime as a BIDS acq_time string (UTC + 'Z', matches mne-bids)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def update_scans_acq_time(bids_root, subject, session, rel_filename, acq_time):
    """Add/update a row in the session scans.tsv with an acq_time datetime.

    Used for the motion file, which mne-bids does not add to scans.tsv. Reads the
    existing scans.tsv (created when the EEG/EMG were written), upserts the row,
    and writes it back. ``acq_time`` is a datetime or None (-> "n/a").
    """
    ses_dir = Path(bids_root) / f"sub-{subject}" / f"ses-{session}"
    scans = ses_dir / f"sub-{subject}_ses-{session}_scans.tsv"
    value = _acq_time_str(acq_time) if acq_time is not None else "n/a"
    rows = {}
    if scans.exists():
        lines = scans.read_text().splitlines()
        for ln in lines[1:]:
            if "\t" in ln:
                name, _, t = ln.partition("\t")
                rows[name] = t
    rows[rel_filename] = value
    scans.parent.mkdir(parents=True, exist_ok=True)
    out = ["filename\tacq_time"] + [f"{n}\t{t}" for n, t in rows.items()]
    scans.write_text("\n".join(out) + "\n")


def apply_eeg_montage(raw, cfg):
    """Apply electrode positions, returning a short status string (never raises).

    * 129-channel recordings use ``cfg.bvef_path`` (numeric-named actiCAP cap).
    * 64-channel recordings either get a renamed cap (``cfg.eeg_rename_labels`` +
      ``cfg.apply_eeg_rename`` -- the unlabeled g.tec case) or standard_1005 when
      their channel names are already real 10-20 labels; otherwise they are left
      generic with no positions.
    """
    try:
        if raw.info["nchan"] == 129 and cfg.bvef_path is not None:
            montage = mne.channels.read_custom_montage(str(cfg.bvef_path))
            raw.set_montage(montage, on_missing="warn")
            return "bvef cap (129 ch)"
        if raw.info["nchan"] == 64:
            if cfg.apply_eeg_rename and cfg.eeg_rename_labels \
                    and len(cfg.eeg_rename_labels) == 64:
                raw.rename_channels(dict(zip(raw.ch_names, cfg.eeg_rename_labels)))
                raw.set_montage("standard_1005", on_missing="warn")
                return "ASSUMED 10-10 order (VERIFY against cabling!)"
            names = set(raw.ch_names)
            std = set(mne.channels.make_standard_montage("standard_1005").ch_names)
            if names & std:
                raw.set_montage("standard_1005", on_missing="warn")
                return "standard_1005"
        return "none (generic labels, no positions)"
    except Exception as e:
        return f"none (montage failed: {type(e).__name__})"


# --------------------------------------------------------------------------- #
# Per-modality conversion
# --------------------------------------------------------------------------- #
def convert_eeg(f, streams, subject, task, session, cfg, recording_dt=None):
    """Write the EEG stream to BIDS. Returns the EEG LSL window (t0, t1).

    EEG is the time reference: its acq_time is the recording datetime itself
    (offset 0); the EMG/mocap files are stamped relative to it.
    """
    eeg = stream_by_type(streams, "EEG")
    if eeg is None:
        raise RuntimeError("no EEG stream")
    eeg_ts = np.asarray(eeg["time_stamps"], float)
    if eeg_ts.size == 0:
        raise RuntimeError("empty EEG stream (0 samples)")
    t0, t1 = float(eeg_ts[0]), float(eeg_ts[-1])

    raw = read_raw_xdf(fname=str(f), stream_ids=[eeg["info"]["stream_id"]],
                       prefix_markers=True)
    status = apply_eeg_montage(raw, cfg)
    raw.info["line_freq"] = cfg.line_freq
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    if recording_dt is not None:
        raw.set_meas_date(recording_dt)  # -> scans.tsv acq_time (EEG reference)

    bids_path = BIDSPath(subject=subject, task=task, session=session,
                         datatype="eeg", root=cfg.bids_root)
    write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True,
                   format="BrainVision", verbose=False)
    print(f"  [EEG]  {raw.info['nchan']} ch, montage: {status}")
    return t0, t1


def convert_emg(f, streams, subject, task, session, eeg_window, cfg, recording_dt=None):
    """Write the EMG stream to BIDS (EDF), cropped to the EEG window if present."""
    emg = stream_by_type(streams, "EMG")
    if emg is None:
        print("  [EMG]  no EMG stream - skipped")
        return False
    emg_ts = np.asarray(emg["time_stamps"], float)
    if emg_ts.size == 0:
        print("  [EMG]  empty EMG stream (0 samples) - skipped")
        return False
    emg_t0 = float(emg_ts[0])
    raw_emg = read_raw_xdf(fname=str(f), stream_ids=[emg["info"]["stream_id"]])

    if eeg_window is not None:
        t0, t1 = eeg_window
        tmin = max(0.0, t0 - emg_t0)
        tmax = min(float(raw_emg.times[-1]), t1 - emg_t0)
        if tmax <= tmin:
            print("  [EMG]  does not overlap EEG window - skipped")
            return False
        raw_emg.crop(tmin=tmin, tmax=tmax)
    else:
        print("  [EMG]  no EEG window - exporting full EMG span (unaligned)")

    present = [c for c in cfg.emg_map if c in raw_emg.ch_names]
    if not present:
        print("  [EMG]  none of the expected channels present - skipped")
        return False
    raw_emg.pick(present)
    raw_emg.rename_channels({c: cfg.emg_map[c] for c in present})
    raw_emg.set_channel_types({ch: "emg" for ch in raw_emg.ch_names})
    raw_emg._data *= cfg.emg_scale  # unit fix (e.g. mV -> V); read_raw_xdf preloads

    srate_emg = round(get_effective_srate_xdf([emg]))
    with raw_emg.info._unlock():
        raw_emg.info["sfreq"] = float(srate_emg)
        raw_emg.info["lowpass"] = srate_emg / 2.0
    raw_emg.info["line_freq"] = cfg.line_freq
    if recording_dt is not None and eeg_window is not None:
        # acq_time = EEG datetime + the EMG start offset within the EEG window
        offset = max(0.0, emg_t0 - eeg_window[0])
        raw_emg.set_meas_date(recording_dt + timedelta(seconds=offset))

    bids_path = BIDSPath(subject=subject, task=task, session=session,
                         datatype="emg", root=cfg.bids_root)
    # EDF/BDF store physical values in 8-char ASCII fields; if EMG amplitude still
    # overflows after scaling it is a residual units issue -> skip+flag.
    try:
        write_raw_bids(raw_emg, bids_path, format="EDF", emg_placement="Other",
                       allow_preload=True, overwrite=True, verbose=False)
    except ValueError as e:
        if "maximum field length" not in str(e):
            raise
        print("  [EMG]  amplitude overflows EDF/BDF's 8-char physical field - "
              "skipped (residual units/calibration issue).")
        return False
    update_sidecar_json(
        bids_path=bids_path.copy().update(suffix="emg", extension=".json"),
        entries={
            "Manufacturer": "Delsys",
            "ManufacturersModelName": "Trigno",
            "EMGReference": "bipolar",
            "EMGGround": "n/a",
            "EMGPlacementSchemeDescription": (
                "Surface electrodes placed according to the SENIAM guidelines "
                "(Surface ElectroMyoGraphy for the Non-Invasive Assessment of Muscles)."
            ),
        },
    )
    print(f"  [EMG]  {len(present)} ch @ {srate_emg} Hz -> {[cfg.emg_map[c] for c in present]}")
    return True


def convert_mocap(streams, subject, task, session, eeg_window, cfg, recording_dt=None):
    """Cluster the mocap stream into markers and write motion BIDS.

    Returns the mocap QC pass/fail flag, or None when there is no usable data.
    """
    mocap = stream_by_type(streams, cfg.mocap_stream_type)
    series = np.asarray(mocap["time_series"]) if mocap is not None else np.empty((0, 4))
    if mocap is None or series.size == 0:
        print("  [MOCAP] no marker data - skipped")
        return None
    ts = np.asarray(mocap["time_stamps"], float)
    if eeg_window is not None:
        t0, t1 = eeg_window
        keep = (ts >= t0) & (ts <= t1)
        if not keep.any():
            print("  [MOCAP] does not overlap EEG window - skipped")
            return None
        series, ts = series[keep], ts[keep]
    else:
        print("  [MOCAP] no EEG window - clustering full mocap span (unaligned)")
    # mocap start offset within the EEG window (for scans.tsv acq_time)
    acq_time = None
    if recording_dt is not None and eeg_window is not None:
        acq_time = recording_dt + timedelta(seconds=float(ts[0]) - eeg_window[0])
    sr = float(mocap["info"]["nominal_srate"][0])

    grid, names, info = cfg.clusterer(series, ts, sr=sr, **cfg.cluster_kwargs)
    n_frames = grid.shape[0]
    n_markers = len(names)

    print(f"  [MOCAP] {series.shape[0]} rows -> {names}")
    for c, name in enumerate(names):
        print(f"          {name:11s}: {info['coverage'][c]:5.1f}% cov, "
              f"max jump {info['max_jump'][c]:.1f} mm")
    print(f"          QC checks: {info['checks']}  all_pass={info['all_checks_pass']}")
    for w in info["warnings"]:
        print(f"          ! {w}")

    # diagnostic plot (saved, not shown)
    t_grid = np.arange(n_frames) / sr
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for c, name in enumerate(names):
        for ax_idx in range(3):
            axes[ax_idx].plot(t_grid, grid[:, c, ax_idx], label=name, alpha=0.8, lw=0.8)
    for ax_idx, label in enumerate(["X (mm)", "Y (mm)", "Z (mm)"]):
        axes[ax_idx].set_ylabel(label)
    axes[0].legend(loc="upper right", fontsize=8, ncol=n_markers)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Clustered marker positions - {subject} {task}")
    plt.tight_layout()
    cfg.qc_dir.mkdir(parents=True, exist_ok=True)
    qc_png = cfg.qc_dir / f"sub-{subject}_ses-{session}_task-{task}_motion-qc.png"
    fig.savefig(qc_png, dpi=90)
    plt.close(fig)

    channels = [
        Channel(channel_name=f"{names[i]}_{axis}", channel_component=axis,
                channel_type="POS", channel_tracked_point=names[i], channel_units="mm")
        for i in range(n_markers) for axis in ["x", "y", "z"]
    ]
    motion = MotionData(
        subject=subject, task_name=task, tracksys=cfg.tracksys,
        sampling_frequency=sr, tracked_points_count=n_markers,
        manufacturer=cfg.manufacturer, recording_type="continuous",
        motion_channel_count=n_markers * 3, recording_duration=n_frames / sr,
        session=session if session else None,
        data=grid.reshape(n_frames, n_markers * 3), channels=channels,
    )
    try:
        validate_motion_data(motion)
    except Exception as e:
        print(f"          [WARN] motion validation: {e}")

    motion_dir = create_bids_directory_structure(
        base_dir=cfg.bids_root, subject=subject, session=session if session else None)
    files = export_bids_motion(motion, out_dir=motion_dir, validate=True, overwrite=True)

    # motionbids writes missing samples as "nan"; BIDS requires "n/a".
    motion_tsv = Path(files["tsv"])
    text = motion_tsv.read_text()
    if "nan" in text:
        motion_tsv.write_text(text.replace("nan", "n/a"))
    # motionbids doesn't touch scans.tsv -> add the motion row with its acq_time
    update_scans_acq_time(cfg.bids_root, subject, session,
                          f"motion/{motion_tsv.name}", acq_time)
    print(f"          [OK] motion BIDS written; QC plot: {qc_png.name}")
    return bool(info["all_checks_pass"])


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def run_conversion(jobs, cfg):
    """Convert a list of ``(xdf_path, subject, session, task, eeg_only)`` jobs.

    Writes the dataset description once, then converts each job with per-modality
    isolation, and prints a batch summary. Returns the per-file summary records.
    """
    make_dataset_description(path=cfg.bids_root, name=cfg.dataset_name, overwrite=True)
    jobs = list(jobs)
    print(f"Discovered {len(jobs)} convertible files")

    summary = []
    for f, subject, session, task, eeg_only in jobs:
        print(f"\n=== {Path(f).name} -> sub-{subject} ses-{session} task-{task}"
              f"{' (EEG only)' if eeg_only else ''} ===")
        rec = {"file": Path(f).name, "subject": subject, "task": task,
               "eeg": False, "emg": None, "motion": None, "errors": {}}
        try:
            streams, header = pyxdf.load_xdf(f, handle_clock_resets=False)
        except Exception as e:
            rec["errors"]["load"] = f"{type(e).__name__}: {e}"
            print(f"  [ERROR] load: {rec['errors']['load']}")
            summary.append(rec)
            continue
        recording_dt = parse_recording_datetime(header)  # EEG-reference acq_time

        eeg_window = None
        try:
            eeg_window = convert_eeg(f, streams, subject, task, session, cfg, recording_dt)
            rec["eeg"] = True
        except Exception as e:
            rec["errors"]["eeg"] = f"{type(e).__name__}: {e}"
            print(f"  [EEG]  [ERROR] {rec['errors']['eeg']}")

        if not eeg_only:
            try:
                rec["emg"] = convert_emg(f, streams, subject, task, session,
                                         eeg_window, cfg, recording_dt)
            except Exception as e:
                rec["errors"]["emg"] = f"{type(e).__name__}: {e}"
                print(f"  [EMG]  [ERROR] {rec['errors']['emg']}")
            try:
                rec["motion"] = convert_mocap(streams, subject, task, session,
                                              eeg_window, cfg, recording_dt)
            except Exception as e:
                rec["errors"]["motion"] = f"{type(e).__name__}: {e}"
                print(f"  [MOCAP] [ERROR] {rec['errors']['motion']}")
        summary.append(rec)

    print_summary(summary)
    return summary


def print_summary(summary):
    """Compact batch report: totals plus anything needing attention."""
    n = len(summary)
    eeg_ok = sum(r["eeg"] for r in summary)
    emg_ok = sum(1 for r in summary if r["emg"])
    motion_ok = sum(1 for r in summary if r["motion"] is not None)
    mocap_pass = sum(1 for r in summary if r["motion"] is True)
    mocap_fail = sum(1 for r in summary if r["motion"] is False)
    errored = [r for r in summary if r["errors"]]
    subjects = sorted({r["subject"] for r in summary})
    print("\n" + "=" * 64)
    print("BATCH SUMMARY")
    print("=" * 64)
    print(f"  files processed : {n}  ({len(subjects)} subjects)")
    print(f"  EEG written     : {eeg_ok}/{n}")
    print(f"  EMG written     : {emg_ok}")
    print(f"  motion written  : {motion_ok}  (QC pass {mocap_pass}, QC fail {mocap_fail})")
    if errored:
        print(f"\n  {len(errored)} file(s) with a per-modality error:")
        for r in errored:
            for mod, msg in r["errors"].items():
                print(f"    - {r['file']} [{mod}]: {msg}")
    mocap_failed = [r for r in summary if r["motion"] is False]
    if mocap_failed:
        print(f"\n  {len(mocap_failed)} file(s) with mocap QC failure (inspect QC plots):")
        for r in mocap_failed:
            print(f"    - {r['file']}")
    if not errored and not mocap_failed:
        print("\n  All files converted; all mocap QC checks passed.")
    print("\nDone.")
