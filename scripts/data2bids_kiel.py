"""Convert the full Kiel T0 dataset to BIDS.

The Kiel recordings live as a flat folder of XDF files named
``data_DEU_<subject>_<session>_<task>.xdf`` (e.g. ``data_DEU_AeSpTZ_T0_ComfSpeed.xdf``).
Stream layout of a walking take:
    * EEG   - "EEG" / "BrainVision RDA", 64 or 129 channels.
    * EMG   - Delsys "DelSys", 9 fixed slots (6 wired to muscles).
    * 6D    - Qualisys mocap, [x, y, z, trajectory_id] rows.
Resting takes carry only the EEG stream; impedance takes carry only an impedance
stream and are skipped.

What gets exported per task:
    * ComfSpeed, FixSpeed, Familiarization -> EEG + EMG + motion (Qualisys).
    * restingstate_sitting/standing, RestState -> EEG only.
    * Impedances, RestState_impedence, any *_old duplicate -> skipped.

Subjects are written site-prefixed as ``sub-Kiel<code>`` in the shared "StepUp"
BIDS root. The EEG/EMG/mocap streams share one LSL clock; the EEG window is used
as the common time base and the EMG/mocap streams are cropped to it.

Mocap clustering uses :func:`src.clustering_tracked.cluster_markers_tracked`
(Qualisys: 1 treadmill + lower-back + sternum stable markers, 2 fragmented feet);
see that module for the rationale.
"""

import matplotlib
matplotlib.use("Agg")  # non-blocking: diagnostic plots are saved, not shown
import matplotlib.pyplot as plt
from mne_bids import make_dataset_description, write_raw_bids, update_sidecar_json, BIDSPath
from mnelab.io.xdf import read_raw_xdf
import mne
import numpy as np
from pathlib import Path
import pyxdf

from motionbids import (
    MotionData,
    Channel,
    export_bids_motion,
    validate_motion_data,
    create_bids_directory_structure,
)

from src.config import DIR_BIDS_ROOT, DIR_PROJ
from src.clustering_tracked import cluster_markers_tracked
from src.xdf import get_effective_srate_xdf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(r"C:\Users\juliu\Desktop\stepup\data\kiel\T0")
DATASET_NAME = "StepUp"      # shared BIDS root (Kiel + Tel Aviv = one multi-site dataset)
SUBJECT_PREFIX = "Kiel"      # code "AeSpTZ" -> sub-KielAeSpTZ
LINE_FREQ = 50               # Hz, mains frequency (required by BIDS)

TRACKSYS = "qualisys"
N_MARKERS = 5  # treadmill, lower back, sternum, left foot, right foot

# The Delsys stream carries no unit; read_raw_xdf assumes Volts, but the values
# are millivolts (Trigno EMG full-scale ~+-11 mV; observed maxima 0.5-11 across
# subjects). Scale mV -> V so the BIDS file stores physically-correct amplitudes
# (and the high-amplitude subjects no longer overflow EDF's 8-char field).
EMG_UNIT_SCALE = 1e-3

# Task routing. Walking tasks get the full export; resting tasks are EEG only.
WALKING_TASKS = {"ComfSpeed", "FixSpeed", "Familiarization"}
# raw task token -> BIDS task label (resting; BIDS recommends "rest*" labels)
RESTING_TASKS = {
    "restingstate_sitting": "restSitting",
    "restingstate_standing": "restStanding",
    "RestState": "rest",
}

# Delsys EMG: 9 fixed slots, 6 wired (slots 4/5/8 are empty). SENIAM placement.
EMG_CHANNEL_MAP = {
    "DelSys_0": "RfEmgR",   # Rectus femoris, right
    "DelSys_1": "RfEmgL",   # Rectus femoris, left
    "DelSys_2": "BfEmgR",   # Biceps femoris, right
    "DelSys_3": "BfEmgL",   # Biceps femoris, left
    "DelSys_6": "GmEmgR",   # Gastrocnemius medialis, right
    "DelSys_7": "GmEmgL",   # Gastrocnemius medialis, left
}


def parse_kiel_filename(stem):
    """Map ``data_DEU_<subject>_<session>_<task>`` to a conversion job.

    Returns ``(subject_label, session, task_label, eeg_only)`` or ``None`` when
    the file should be skipped (impedance check, *_old duplicate, unrecognised).
    """
    parts = stem.split("_")
    if len(parts) < 5 or parts[0] != "data" or parts[1] != "DEU":
        return None
    code = "".join(c for c in parts[2] if c.isalnum())  # strip spaces / oddities
    session = "".join(c for c in parts[3] if c.isalnum())
    task = "_".join(parts[4:])

    low = task.lower()
    if "impedance" in low or "impedence" in low or "_old" in low:
        return None  # impedance check or superseded duplicate
    if task in WALKING_TASKS:
        return SUBJECT_PREFIX + code, session, task, False
    if task in RESTING_TASKS:
        return SUBJECT_PREFIX + code, session, RESTING_TASKS[task], True
    return None  # unrecognised task token


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def stream_by_type(streams, stype):
    matches = [s for s in streams if s["info"]["type"][0] == stype]
    return matches[0] if matches else None


def apply_eeg_montage(raw):
    """Apply electrode positions, returning a short status string (never raises).

    129-channel recordings use the project .bvef cap file; 64-channel recordings
    use standard_1005 when their channel names are real 10-20 labels, otherwise
    the channels are left generic with no positions.
    """
    try:
        if raw.info["nchan"] == 129:
            bvef = DIR_PROJ.joinpath("src", "montages", "stepup_kiel.bvef")
            raw.set_montage(mne.channels.read_custom_montage(str(bvef)), on_missing="warn")
            return "stepup_kiel.bvef (129 ch)"
        if raw.info["nchan"] == 64:
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
def convert_eeg(f, streams, subject, task, session):
    """Write the EEG stream to BIDS. Returns the EEG LSL window (t0, t1)."""
    eeg = stream_by_type(streams, "EEG")
    if eeg is None:
        raise RuntimeError("no EEG stream")
    eeg_ts = np.asarray(eeg["time_stamps"], float)
    if eeg_ts.size == 0:
        raise RuntimeError("empty EEG stream (0 samples)")
    t0, t1 = float(eeg_ts[0]), float(eeg_ts[-1])

    raw = read_raw_xdf(fname=str(f), stream_ids=[eeg["info"]["stream_id"]],
                       prefix_markers=True)
    status = apply_eeg_montage(raw)
    raw.info["line_freq"] = LINE_FREQ
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    bids_path = BIDSPath(subject=subject, task=task, session=session,
                         datatype="eeg", root=DIR_BIDS_ROOT)
    write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True,
                   format="BrainVision", verbose=False)
    print(f"  [EEG]  {raw.info['nchan']} ch, montage: {status}")
    return t0, t1


def convert_emg(f, streams, subject, task, session, eeg_window):
    """Write the EMG stream to BIDS (EDF), cropped to the EEG window if present."""
    emg = stream_by_type(streams, "EMG")
    if emg is None:
        print("  [EMG]  no EMG stream - skipped")
        return False
    emg_ts = np.asarray(emg["time_stamps"], float)
    if emg_ts.size == 0:
        print("  [EMG]  empty EMG stream (0 samples) - skipped")
        return False
    raw_emg = read_raw_xdf(fname=str(f), stream_ids=[emg["info"]["stream_id"]])

    if eeg_window is not None:
        t0, t1 = eeg_window
        emg_t0 = float(emg_ts[0])
        tmin = max(0.0, t0 - emg_t0)
        tmax = min(float(raw_emg.times[-1]), t1 - emg_t0)
        if tmax <= tmin:
            print("  [EMG]  does not overlap EEG window - skipped")
            return False
        raw_emg.crop(tmin=tmin, tmax=tmax)
    else:
        print("  [EMG]  no EEG window - exporting full EMG span (unaligned)")

    present = [c for c in EMG_CHANNEL_MAP if c in raw_emg.ch_names]
    if not present:
        print("  [EMG]  none of the expected DelSys channels present - skipped")
        return False
    raw_emg.pick(present)
    raw_emg.rename_channels({c: EMG_CHANNEL_MAP[c] for c in present})
    raw_emg.set_channel_types({ch: "emg" for ch in raw_emg.ch_names})
    raw_emg._data *= EMG_UNIT_SCALE  # mV -> V (read_raw_xdf returns preloaded data)

    srate_emg = round(get_effective_srate_xdf([emg]))
    with raw_emg.info._unlock():
        raw_emg.info["sfreq"] = float(srate_emg)
        raw_emg.info["lowpass"] = srate_emg / 2.0
    raw_emg.info["line_freq"] = LINE_FREQ

    bids_path = BIDSPath(subject=subject, task=task, session=session,
                         datatype="emg", root=DIR_BIDS_ROOT)
    # EDF/BDF store physical values in 8-char ASCII fields; large-amplitude EMG
    # (|value| >= 10 in the stream unit -> >= 1e7 uV) overflows that field and
    # BIDS emg allows only EDF/BDF, so skip+flag rather than guess a rescaling.
    try:
        write_raw_bids(raw_emg, bids_path, format="EDF", emg_placement="Other",
                       allow_preload=True, overwrite=True, verbose=False)
    except ValueError as e:
        if "maximum field length" not in str(e):
            raise
        print("  [EMG]  amplitude overflows EDF/BDF's 8-char physical field "
              "- skipped (units/calibration issue; resolve upstream).")
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
    print(f"  [EMG]  {len(present)} ch @ {srate_emg} Hz -> {[EMG_CHANNEL_MAP[c] for c in present]}")
    return True


def convert_mocap(streams, subject, task, session, eeg_window, qc_dir):
    """Cluster the Qualisys 6D stream into 5 markers and write motion BIDS."""
    mocap = stream_by_type(streams, "6D")
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
    sr = float(mocap["info"]["nominal_srate"][0])

    grid, names, info = cluster_markers_tracked(series, ts, sr=sr, n_markers=N_MARKERS)
    n_frames = grid.shape[0]
    n_channels = N_MARKERS * 3

    print(f"  [MOCAP] {series.shape[0]} rows, high IDs {info['high_marker_ids']} "
          f"foot IDs {info['foot_ids']}")
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
    axes[0].legend(loc="upper right", fontsize=8, ncol=N_MARKERS)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Clustered marker positions - {subject} {task}")
    plt.tight_layout()
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_png = qc_dir / f"sub-{subject}_ses-{session}_task-{task}_motion-qc.png"
    fig.savefig(qc_png, dpi=90)
    plt.close(fig)

    channels = [
        Channel(channel_name=f"{names[i]}_{axis}", channel_component=axis,
                channel_type="POS", channel_tracked_point=names[i], channel_units="mm")
        for i in range(N_MARKERS) for axis in ["x", "y", "z"]
    ]
    motion = MotionData(
        subject=subject, task_name=task, tracksys=TRACKSYS,
        sampling_frequency=sr, tracked_points_count=N_MARKERS,
        manufacturer="Qualisys", recording_type="continuous",
        motion_channel_count=n_channels, recording_duration=n_frames / sr,
        session=session if session else None,
        data=grid.reshape(n_frames, n_channels), channels=channels,
    )
    try:
        validate_motion_data(motion)
    except Exception as e:
        print(f"          [WARN] motion validation: {e}")

    motion_dir = create_bids_directory_structure(
        base_dir=DIR_BIDS_ROOT, subject=subject, session=session if session else None)
    files = export_bids_motion(motion, out_dir=motion_dir, validate=True, overwrite=True)

    # motionbids writes missing samples as "nan"; BIDS requires "n/a".
    motion_tsv = Path(files["tsv"])
    text = motion_tsv.read_text()
    if "nan" in text:
        motion_tsv.write_text(text.replace("nan", "n/a"))
    print(f"          [OK] motion BIDS written; QC plot: {qc_png.name}")
    return bool(info["all_checks_pass"])


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def _discover_files():
    """Yield ``(xdf_path, subject, session, task, eeg_only)`` for convertible files."""
    for f in sorted(DATA_DIR.glob("*.xdf")):
        job = parse_kiel_filename(f.stem)
        if job is None:
            continue
        subject, session, task, eeg_only = job
        yield f, subject, session, task, eeg_only


def main(subjects=None):
    """Convert the Kiel T0 dataset. ``subjects`` optionally limits to a set of
    subject labels (e.g. {"KielAeSpTZ"}) for verification / parallel chunking."""
    if not DATA_DIR.is_dir():
        print(f"Data dir not found: {DATA_DIR}")
        return []
    make_dataset_description(path=DIR_BIDS_ROOT, name=DATASET_NAME, overwrite=True)
    qc_dir = Path(DIR_BIDS_ROOT) / "derivatives" / "qc"

    jobs = [j for j in _discover_files() if subjects is None or j[1] in subjects]
    print(f"Discovered {len(jobs)} convertible files under {DATA_DIR}")

    summary = []
    for f, subject, session, task, eeg_only in jobs:
        print(f"\n=== {f.name} -> sub-{subject} ses-{session} task-{task}"
              f"{' (EEG only)' if eeg_only else ''} ===")
        rec = {"file": f.name, "subject": subject, "task": task,
               "eeg": False, "emg": None, "motion": None, "errors": {}}
        try:
            streams, _ = pyxdf.load_xdf(f, handle_clock_resets=False)
        except Exception as e:
            rec["errors"]["load"] = f"{type(e).__name__}: {e}"
            print(f"  [ERROR] load: {rec['errors']['load']}")
            summary.append(rec)
            continue

        eeg_window = None
        try:
            eeg_window = convert_eeg(f, streams, subject, task, session)
            rec["eeg"] = True
        except Exception as e:
            rec["errors"]["eeg"] = f"{type(e).__name__}: {e}"
            print(f"  [EEG]  [ERROR] {rec['errors']['eeg']}")

        if not eeg_only:
            try:
                rec["emg"] = convert_emg(f, streams, subject, task, session, eeg_window)
            except Exception as e:
                rec["errors"]["emg"] = f"{type(e).__name__}: {e}"
                print(f"  [EMG]  [ERROR] {rec['errors']['emg']}")
            try:
                rec["motion"] = convert_mocap(streams, subject, task, session,
                                              eeg_window, qc_dir)
            except Exception as e:
                rec["errors"]["motion"] = f"{type(e).__name__}: {e}"
                print(f"  [MOCAP] [ERROR] {rec['errors']['motion']}")
        summary.append(rec)

    _print_summary(summary)
    return summary


def _print_summary(summary):
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


if __name__ == "__main__":
    main()
