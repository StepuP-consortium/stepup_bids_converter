"""Tel Aviv-site configuration and entry point for XDF -> BIDS conversion.

Tel Aviv recordings are nested as ``<root>/<subject>/<session>/<task>.xdf``.
Walking takes carry g.tec EEG (64 ch) + Delsys EMG + Vicon mocap; sitting/standing
carry only EEG (exported EEG-only by request). See :mod:`src.bids_convert` for
the engine and :func:`src.clustering_vicon.cluster_markers_vicon` for the mocap
clustering.

EEG note: the g.tec stream carries no channel labels (GTec_0..63). No authoritative
g.tec index->electrode mapping is published (the cap is freely re-cabled), so EEG is
exported with generic labels by default. The candidate canonical 10-10 order sits in
``GTEC_64_LABELS`` behind ``APPLY_GTEC_MONTAGE`` -- enable only after confirming it
matches the cabling.

Run via ``scripts/data2bids_telaviv.py`` (a thin launcher) or ``bids_telaviv.main()``.
"""

from __future__ import annotations

from pathlib import Path

from .config import DIR_BIDS_ROOT
from .clustering_vicon import cluster_markers_vicon
from .bids_convert import SiteConfig, run_conversion

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(r"C:\Users\juliu\Desktop\stepup\data\tel_aviv")
SESSIONS = ["T0"]            # set ["T0","T1","T2"] for the full longitudinal dataset
SUBJECT_PREFIX = "TelAviv"   # folder "001" -> sub-TelAviv001
DATASET_NAME = "StepUp"
LINE_FREQ = 50               # Hz

# File stem (lower-cased, trailing dots stripped) -> (BIDS task label, eeg_only).
TASK_LABELS = {
    "sitting": ("Sitting", True),
    "standing": ("Standing", True),
    "walking1": ("FAM", False),
    "walking2": ("CS", False),
    "walking3": ("FS", False),
    "walking": ("Walking", False),  # unnumbered walking trial (e.g. 014/T0)
}

# Delsys EMG: 8 slots. SENIAM placement.
EMG_CHANNEL_MAP = {
    "DelsysData_0": "RfEmgL", "DelsysData_1": "RfEmgR",
    "DelsysData_2": "BfEmgR", "DelsysData_3": "BfEmgL",
    "DelsysData_4": "GaEmgR", "DelsysData_5": "GaEmgL",
    "DelsysData_6": "TaEmgR", "DelsysData_7": "TaEmgL",
}
# Delsys is unit-less in the stream; values are millivolts (Trigno full-scale
# ~+-11 mV). Scale mV -> V for correct BIDS amplitudes.
EMG_UNIT_SCALE = 1e-3

# --- EEG montage (see module docstring) ---
APPLY_GTEC_MONTAGE = False
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


def parse_task(stem):
    """Map an XDF file stem to ``(bids_task_label, eeg_only)`` or ``None``.

    Case-insensitive and tolerant of a trailing-dot typo. Unknown stems fall
    back to a sanitized (alphanumeric) label with a full export.
    """
    key = stem.strip().lower().rstrip(".")
    if key in TASK_LABELS:
        return TASK_LABELS[key]
    return ("".join(ch for ch in stem if ch.isalnum()) or "unknown", False)


def discover_files():
    """Yield ``(xdf_path, subject, session, task, eeg_only)`` for the scope."""
    for subj_dir in sorted(d for d in DATA_ROOT.iterdir() if d.is_dir()):
        subject = SUBJECT_PREFIX + subj_dir.name
        for session in SESSIONS:
            ses_dir = subj_dir / session
            if not ses_dir.is_dir():
                continue
            for f in sorted(ses_dir.glob("*.xdf")):
                task, eeg_only = parse_task(f.stem)
                yield f, subject, session, task, eeg_only


def build_config():
    return SiteConfig(
        bids_root=DIR_BIDS_ROOT,
        dataset_name=DATASET_NAME,
        qc_dir=Path(DIR_BIDS_ROOT) / "derivatives" / "qc",
        line_freq=LINE_FREQ,
        bvef_path=None,
        eeg_rename_labels=GTEC_64_LABELS,
        apply_eeg_rename=APPLY_GTEC_MONTAGE,
        emg_map=EMG_CHANNEL_MAP,
        emg_scale=EMG_UNIT_SCALE,
        mocap_stream_type="MoCap",
        clusterer=cluster_markers_vicon,
        cluster_kwargs={},
        tracksys="vicon",
        manufacturer="Vicon",
    )


def main(subjects=None):
    """Convert the Tel Aviv dataset. ``subjects`` optionally limits to a set of
    subject labels (e.g. {"TelAviv001"}) for verification."""
    if not DATA_ROOT.is_dir():
        print(f"Data root not found: {DATA_ROOT}")
        return []
    jobs = [j for j in discover_files() if subjects is None or j[1] in subjects]
    return run_conversion(jobs, build_config())
