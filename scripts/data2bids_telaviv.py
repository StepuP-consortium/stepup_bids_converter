"""Convert all Tel Aviv T0 recordings to BIDS.

Layout of a Tel Aviv take (verified for sub-TelAviv001 / T0):
    * EEG   - g.tec "GTec", 64 channels @ 250 Hz. The LSL stream carries no
              channel labels, so channels arrive as GTec_0..GTec_63 (see the
              montage note below).
    * EMG   - Delsys "DelsysData", 8 channels @ ~2148 Hz.
    * MoCap - Vicon "Vicon", [x, y, z, trajectory_id] rows @ 100 Hz. Empty for
              the Sitting take.

What gets exported per task (per user request):
    * Sitting, Standing  -> EEG only.
    * Walking1/2/3        -> EEG + EMG + motion (Vicon).
Walking file names are mapped to task labels: Walking1->FAM, Walking2->CS,
Walking3->FS.

The EEG, EMG and mocap streams share one LSL clock but cover different spans;
the EEG is the shortest, so its timestamp range is used as the common window and
the EMG / mocap streams are cropped to it before conversion.

EEG montage note
----------------
The g.tec stream does not transmit electrode labels, so we cannot know which
amplifier channel maps to which 10-05 electrode without the lab's cabling/montage
configuration. No authoritative g.tec index->label file is published (the
g.Nautilus RESEARCH cap is freely re-cabled per lab). We therefore export the
EEG with honest generic labels (GTec_0..63) and no positions by default. The
canonical 64-channel 10-10 order is provided below as ``GTEC_64_LABELS``; set
``APPLY_GTEC_MONTAGE = True`` ONLY after confirming it matches the Tel Aviv
cabling, which renames the channels and applies standard_1005 positions.
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
from src.clustering_vicon import cluster_markers_vicon
from src.xdf import get_effective_srate_xdf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(r"C:\Users\juliu\Desktop\stepup\data\tel_aviv")
# Sessions to convert. Per request only the baseline (T0); set to
# ["T0", "T1", "T2"] to convert the full longitudinal dataset.
SESSIONS = ["T0"]
SUBJECT_PREFIX = "TelAviv"  # folder "001" -> sub-TelAviv001 (site-prefixed in the shared root)
DATASET_NAME = "StepUp"     # shared BIDS root (Kiel + Tel Aviv = one multi-site dataset)

# File stem (lower-cased, trailing dots stripped) -> (BIDS task label, eeg_only).
# Sitting/Standing export EEG only; the walking trials add EMG + motion.
# File names vary in case across subjects (Sitting vs sitting), and a few are
# irregular (an unnumbered "walking", a typo'd "standing.."), so matching is
# done case-insensitively after stripping trailing dots.
TASK_LABELS = {
    "sitting": ("Sitting", True),
    "standing": ("Standing", True),
    "walking1": ("FAM", False),
    "walking2": ("CS", False),
    "walking3": ("FS", False),
    "walking": ("Walking", False),  # unnumbered walking trial (e.g. 014/T0)
}


def parse_task(stem):
    """Map an XDF file stem to ``(bids_task_label, eeg_only)``.

    Case-insensitive and tolerant of a trailing-dot typo. Unknown stems fall
    back to a sanitized (alphanumeric) version of the stem with a full export.
    """
    key = stem.strip().lower().rstrip(".")
    if key in TASK_LABELS:
        return TASK_LABELS[key]
    return ("".join(ch for ch in stem if ch.isalnum()) or "unknown", False)

LINE_FREQ = 50  # Hz, mains frequency (required by BIDS)
# Delsys streams carry no unit; read_raw_xdf assumes Volts, but the values are
# millivolts (Trigno EMG full-scale ~+-11 mV). Scale mV -> V so the BIDS file
# stores physically-correct amplitudes and high-amplitude subjects no longer
# overflow EDF's 8-char physical field.
EMG_UNIT_SCALE = 1e-3

# g.tec EMG (Delsys) channel map. The stream exposes 8 fixed slots; 4/5 are of
# unknown wiring here and kept as placeholders so no data is silently dropped.
# Electrode placement follows the SENIAM guidelines.
EMG_CHANNEL_MAP = {
    "DelsysData_0": "RfEmgR",   # Rectus femoris, right
    "DelsysData_1": "RfEmgL",   # Rectus femoris, left
    "DelsysData_2": "BfEmgR",   # Biceps femoris, right
    "DelsysData_3": "BfEmgL",   # Biceps femoris, left
    "DelsysData_4": "tbd1",     # wiring unknown
    "DelsysData_5": "tbd2",     # wiring unknown
    "DelsysData_6": "GmEmgR",   # Gastrocnemius medialis, right
    "DelsysData_7": "GmEmgL",   # Gastrocnemius medialis, left
}

# --- EEG montage (see module docstring) ---
APPLY_GTEC_MONTAGE = False  # set True ONLY after confirming the order matches the cabling
# Candidate canonical 64-channel 10-10 order. UNVERIFIED against the Tel Aviv
# cabling - confirm before enabling APPLY_GTEC_MONTAGE.
GTEC_64_LABELS = [
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    "F9", "F10", "TP9", "TP10",
]

TRACKSYS = "vicon"
N_MARKERS = 5  # lower back + left/right heel + left/right toe


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def stream_by_type(streams, stype):
    """Return the first stream of the given LSL type, or None."""
    matches = [s for s in streams if s["info"]["type"][0] == stype]
    return matches[0] if matches else None


def apply_eeg_montage(raw):
    """Apply electrode positions to an EEG Raw, returning a short status string.

    129-channel recordings use the project .bvef cap file. 64-channel g.tec
    recordings have anonymous channel names; positions are applied only when
    ``APPLY_GTEC_MONTAGE`` is set and the channel count matches the candidate
    label list (see the module docstring), otherwise the channels are left
    generic with no positions.
    """
    if raw.info["nchan"] == 129:
        bvef = DIR_PROJ.joinpath("utils", "montages", "stepup_kiel.bvef")
        raw.set_montage(mne.channels.read_custom_montage(str(bvef)))
        return "stepup_kiel.bvef (129 ch)"
    if raw.info["nchan"] == 64 and APPLY_GTEC_MONTAGE:
        if len(GTEC_64_LABELS) != 64:
            return "GTEC_64_LABELS is not 64 long - left channels generic"
        raw.rename_channels(dict(zip(raw.ch_names, GTEC_64_LABELS)))
        raw.set_montage("standard_1005", on_missing="warn")
        return "ASSUMED canonical 10-10 order (VERIFY against cabling!)"
    # default: anonymous channels, no positions
    if raw.info["nchan"] == 64:
        names = set(raw.ch_names)
        std = set(mne.channels.make_standard_montage("standard_1005").ch_names)
        if names & std:  # real 10-20 names already present (e.g. a different site)
            raw.set_montage("standard_1005", on_missing="warn")
            return "standard_1005 (matched existing labels)"
    return "none (generic labels, no positions)"


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
    """Write the EMG stream to BIDS (EDF), cropped to the EEG window."""
    emg = stream_by_type(streams, "EMG")
    if emg is None:
        print("  [EMG]  no EMG stream - skipped")
        return False
    emg_ts = np.asarray(emg["time_stamps"], float)
    if emg_ts.size == 0:
        print("  [EMG]  empty EMG stream (0 samples) - skipped")
        return False
    raw_emg = read_raw_xdf(fname=str(f), stream_ids=[emg["info"]["stream_id"]])

    # Crop to the EEG window when there is one; otherwise (no EEG to anchor to)
    # keep the full EMG span so the data is not lost.
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
    # EDF/BDF store physical values in 8-char ASCII fields. MNE converts the
    # (unit-less, so assumed-Volt) stream to uV on export, so a stream value of
    # ~+/-11 becomes ~1.1e7 uV and overflows the field. BIDS emg allows only
    # EDF/BDF, so there is no format that avoids this. Such amplitudes (+/-11 V)
    # are non-physiological for EMG (a units/calibration issue in the recording),
    # so skip+flag rather than guess a rescaling that would desync the dataset.
    try:
        write_raw_bids(raw_emg, bids_path, format="EDF", emg_placement="Other",
                       allow_preload=True, overwrite=True, verbose=False)
    except ValueError as e:
        if "maximum field length" not in str(e):
            raise
        print("  [EMG]  amplitude overflows EDF/BDF's 8-char physical field "
              "(stream ~+/-11 -> ~1.1e7 uV) - skipped. Likely a units/calibration "
              "issue in this recording; resolve upstream or choose a rescaling.")
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
    print(f"  [EMG]  {len(present)} ch @ {srate_emg} Hz [EDF] -> "
          f"{[EMG_CHANNEL_MAP[c] for c in present]}")
    return True


def convert_mocap(streams, subject, task, session, eeg_window, qc_dir):
    """Cluster the Vicon stream into 5 markers and write motion BIDS."""
    mocap = stream_by_type(streams, "MoCap")
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

    grid, names, info = cluster_markers_vicon(series, ts, sr=sr)
    n_frames = grid.shape[0]
    n_channels = N_MARKERS * 3

    print(f"  [MOCAP] {series.shape[0]} rows, max simultaneous foot rows "
          f"{info['n_foot_candidates_max']} (expect 4)")
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
        manufacturer="Vicon", recording_type="continuous",
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
    """Yield ``(xdf_path, subject_label, session, task, eeg_only)`` for the scope."""
    for subj_dir in sorted(d for d in DATA_ROOT.iterdir() if d.is_dir()):
        subject = SUBJECT_PREFIX + subj_dir.name
        for session in SESSIONS:
            ses_dir = subj_dir / session
            if not ses_dir.is_dir():
                continue
            for f in sorted(ses_dir.glob("*.xdf")):
                task, eeg_only = parse_task(f.stem)
                yield f, subject, session, task, eeg_only


def main():
    if not DATA_ROOT.is_dir():
        print(f"Data root not found: {DATA_ROOT}")
        return
    make_dataset_description(path=DIR_BIDS_ROOT, name=DATASET_NAME, overwrite=True)
    qc_dir = Path(DIR_BIDS_ROOT) / "derivatives" / "qc"

    jobs = list(_discover_files())
    print(f"Discovered {len(jobs)} files under {DATA_ROOT} (sessions={SESSIONS})")

    summary = []  # one record per file
    for f, subject, session, task, eeg_only in jobs:
        rel = f"{f.parent.parent.name}/{session}/{f.name}"
        print(f"\n=== {rel} -> sub-{subject} ses-{session} task-{task}"
              f"{' (EEG only)' if eeg_only else ''} ===")
        rec = {"file": rel, "subject": subject, "task": task,
               "eeg": False, "emg": None, "motion": None, "errors": {}}

        try:
            streams, _ = pyxdf.load_xdf(f, handle_clock_resets=False)
        except Exception as e:
            rec["errors"]["load"] = f"{type(e).__name__}: {e}"
            print(f"  [ERROR] load: {rec['errors']['load']}")
            summary.append(rec)
            continue

        # Each modality is isolated: a failure in one does not block the others.
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


def _print_summary(summary):
    """Compact batch report: totals plus anything needing attention."""
    n = len(summary)
    eeg_ok = sum(r["eeg"] for r in summary)
    emg_ok = sum(1 for r in summary if r["emg"])
    motion_ok = sum(1 for r in summary if r["motion"] is not None)
    mocap_pass = sum(1 for r in summary if r["motion"] is True)
    mocap_fail = sum(1 for r in summary if r["motion"] is False)
    errored = [r for r in summary if r["errors"]]
    print("\n" + "=" * 64)
    print("BATCH SUMMARY")
    print("=" * 64)
    print(f"  files processed : {n}")
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
