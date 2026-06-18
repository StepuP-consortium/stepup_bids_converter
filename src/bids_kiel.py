"""Kiel-site configuration and entry point for XDF -> BIDS conversion.

Kiel recordings are a flat folder of XDF files named
``data_DEU_<subject>_<session>_<task>.xdf``. Walking takes carry EEG (64 or 129
ch) + Delsys EMG + Qualisys 6D mocap; resting takes carry only EEG; impedance
takes are skipped. See :mod:`src.bids_convert` for the conversion engine and
:func:`src.clustering_tracked.cluster_markers_tracked` for the mocap clustering.

Run via ``scripts/data2bids_kiel.py`` (a thin launcher) or ``bids_kiel.main()``.
"""

from __future__ import annotations

from pathlib import Path

from .config import DIR_BIDS_ROOT, DIR_PROJ
from .clustering_tracked import cluster_markers_tracked
from .bids_convert import SiteConfig, run_conversion

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(r"C:\Users\juliu\Desktop\stepup\data\kiel\T0")
DATASET_NAME = "StepUp"      # shared BIDS root (Kiel + Tel Aviv = one multi-site dataset)
SUBJECT_PREFIX = "Kiel"      # code "AeSpTZ" -> sub-KielAeSpTZ
LINE_FREQ = 50               # Hz
N_MARKERS = 5                # treadmill, lower back, sternum, left foot, right foot

# Task routing. Walking tasks get the full export; resting tasks are EEG only.
WALKING_TASKS = {"ComfSpeed", "FixSpeed", "Familiarization"}
RESTING_TASKS = {
    "restingstate_sitting": "restSitting",
    "restingstate_standing": "restStanding",
    "RestState": "rest",
}

# Delsys EMG: 9 fixed slots, 6 wired (slots 4/5/8 empty). SENIAM placement.
EMG_CHANNEL_MAP = {
    "DelSys_0": "RfEmgR", "DelSys_1": "RfEmgL",
    "DelSys_2": "BfEmgR", "DelSys_3": "BfEmgL",
    "DelSys_6": "GmEmgR", "DelSys_7": "GmEmgL",
}
# The Delsys stream is unit-less; read_raw_xdf assumes Volts but the values are
# millivolts (Trigno EMG full-scale ~+-11 mV). Scale mV -> V for correct BIDS.
EMG_UNIT_SCALE = 1e-3

BVEF_PATH = DIR_PROJ.joinpath("src", "montages", "stepup_kiel.bvef")  # 129-ch cap


def parse_kiel_filename(stem):
    """Map ``data_DEU_<subject>_<session>_<task>`` to a conversion job.

    Returns ``(subject_label, session, task_label, eeg_only)`` or ``None`` when
    the file is skipped (impedance check, *_old duplicate, unrecognised task).
    """
    parts = stem.split("_")
    if len(parts) < 5 or parts[0] != "data" or parts[1] != "DEU":
        return None
    code = "".join(c for c in parts[2] if c.isalnum())  # strip spaces / oddities
    session = "".join(c for c in parts[3] if c.isalnum())
    task = "_".join(parts[4:])

    low = task.lower()
    if "impedance" in low or "impedence" in low or "_old" in low:
        return None
    if task in WALKING_TASKS:
        return SUBJECT_PREFIX + code, session, task, False
    if task in RESTING_TASKS:
        return SUBJECT_PREFIX + code, session, RESTING_TASKS[task], True
    return None


def discover_files():
    """Yield ``(xdf_path, subject, session, task, eeg_only)`` for convertible files."""
    for f in sorted(DATA_DIR.glob("*.xdf")):
        job = parse_kiel_filename(f.stem)
        if job is not None:
            yield (f, *job)


def build_config():
    return SiteConfig(
        bids_root=DIR_BIDS_ROOT,
        dataset_name=DATASET_NAME,
        qc_dir=Path(DIR_BIDS_ROOT) / "derivatives" / "qc",
        line_freq=LINE_FREQ,
        bvef_path=BVEF_PATH,
        emg_map=EMG_CHANNEL_MAP,
        emg_scale=EMG_UNIT_SCALE,
        mocap_stream_type="6D",
        clusterer=cluster_markers_tracked,
        cluster_kwargs={"n_markers": N_MARKERS},
        tracksys="qualisys",
        manufacturer="Qualisys",
    )


def main(subjects=None):
    """Convert the Kiel dataset. ``subjects`` optionally limits to a set of
    subject labels (e.g. {"KielAeSpTZ"}) for verification."""
    if not DATA_DIR.is_dir():
        print(f"Data dir not found: {DATA_DIR}")
        return []
    jobs = [j for j in discover_files() if subjects is None or j[1] in subjects]
    return run_conversion(jobs, build_config())
