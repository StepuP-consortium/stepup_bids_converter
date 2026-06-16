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

##################################
# Create a dataset description
##################################
# This is a required step for BIDS datasets
# It creates a dataset_description.json file in the BIDS root directory
make_dataset_description(path=DIR_BIDS_ROOT, name='StepUp_Kiel')

# Set file path to the directory containing the XDF files (source/Kiel*/.../*DEU*.xdf)
file_path = Path(r"C:\Users\juliu\Desktop\kiel\stepup_bids_converter\data\source")  # Replace with your XDF file path)

# find all DEU .xdf files in the Kiel subject folders (source/Kiel*/.../*DEU*.xdf)
xdf_files = list(file_path.glob('Kiel*/**/*DEU*.xdf'))
if not xdf_files:
    print("No XDF files found in the directory.")
else:
    for xdf_file in xdf_files:
        print(f"Found XDF file: {xdf_file.name}")

# load the first xdf file
streams, fileheader = pyxdf.load_xdf(xdf_files[0], handle_clock_resets=False)
print("File loaded successfully.")

if streams is not None:
    print(f"Number of streams: {len(streams)}")
    for i, stream in enumerate(streams):
        print(f"\nStream {i+1}:")
        print(f"Name: {stream['info']['name'][0]}")
        print(f"Type: {stream['info']['type'][0]}")
        print(f"Channel count: {stream['info']['channel_count'][0]}")
        print(f"Sample rate: {stream['info']['nominal_srate'][0]}")
        print(f"Data points: {len(stream['time_series'])}")

fname = xdf_files[0].name
SUBJECT_ID = fname.split('_')[2]  # Extract subject ID from filename
TASK = fname.split('_')[-1].split('.')[0]  # Extract task from filename
SESSION = fname.split('_')[3]  # Extract visit from filename


###################################
# EEG 2 BIDS
###################################

# load EEG stream into MNE and export to BIDS
eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
eeg_stream_id = eeg_stream['info']['stream_id']

raw = read_raw_xdf(fname=file_path.joinpath(xdf_files[0]), stream_ids=[eeg_stream_id], prefix_markers=True)  # this is a mne.io.Raw object

# set montage using the provided .bvef file for channel locations
if raw.info['nchan'] == 64:
    raw.set_montage('standard_1005')  # fallback to standard 10-20 montage if channel count doesn't match the .bvef
elif raw.info['nchan'] == 129:
    bvef_path = DIR_PROJ.joinpath('utils', 'montages', 'stepup_kiel.bvef')
    montage = mne.channels.read_custom_montage(str(bvef_path))
    raw.set_montage(montage)

# set info for BIDS
raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

# write raw data to BIDS
bids_path_eeg = BIDSPath(subject=SUBJECT_ID, task=TASK, session=SESSION, datatype='eeg', root=DIR_BIDS_ROOT)
write_raw_bids(raw, bids_path_eeg, overwrite=True, allow_preload=True, format='BrainVision', verbose=True)

print(f'Finished writing EEG BIDS for participant {SUBJECT_ID} and task {TASK}')




###################################
# EMG 2 BIDS
###################################
# EMG was recorded with a Delsys Trigno system (stream type 'EMG').
# Load it into MNE the same way as the EEG stream above, then write it to BIDS
# with write_raw_bids using the dedicated 'emg' datatype (as in the MNE-BIDS
# FieldTrip EMG example: build a Raw, mark channels as 'emg', export as EDF).
emg_stream = [s for s in streams if s['info']['type'][0] == 'EMG'][0]
emg_stream_id = emg_stream['info']['stream_id']

raw_emg = read_raw_xdf(fname=file_path.joinpath(xdf_files[0]), stream_ids=[emg_stream_id])  # mne.io.Raw

# The Delsys stream exposes 9 fixed slots but only 6 are wired to a sensor
# (DelSys_4, DelSys_5 and DelSys_8 are empty). Map the active channels to their
# muscle/side; electrode placement follows the SENIAM guidelines:
#   electrode 2/3 -> Rectus femoris (right/left)
#   electrode 4/5 -> Biceps femoris (right/left)
#   electrode 6/7 -> Gastrocnemius medialis (right/left)
emg_channel_map = {
    'DelSys_0': 'RfEmgR',   # Rectus femoris, right
    'DelSys_1': 'RfEmgL',   # Rectus femoris, left
    'DelSys_2': 'BfEmgR',   # Biceps femoris, right
    'DelSys_3': 'BfEmgL',   # Biceps femoris, left
    'DelSys_6': 'GmEmgR',   # Gastrocnemius medialis, right
    'DelSys_7': 'GmEmgL',   # Gastrocnemius medialis, left
}

# keep only the wired channels, rename them, and tag them as EMG
raw_emg.pick(list(emg_channel_map.keys()))
raw_emg.rename_channels(emg_channel_map)
raw_emg.set_channel_types({ch: 'emg' for ch in raw_emg.ch_names})

# Use the effective sample rate from the EMG stream (derived from the LSL
# timestamps), not the nominal rate. EDF stores integer sample rates, so round
# it (the sub-Hz remainder is negligible) and let MNE pad the final partial
# data record.
srate_emg = round(get_effective_srate_xdf([emg_stream]))
with raw_emg.info._unlock():
    raw_emg.info['sfreq'] = float(srate_emg)
    raw_emg.info['lowpass'] = srate_emg / 2.0

raw_emg.info['line_freq'] = 50  # power line frequency, required by BIDS

# write EMG to BIDS as EDF (BIDS allows BDF or EDF for the 'emg' datatype)
bids_path_emg = BIDSPath(subject=SUBJECT_ID, task=TASK, session=SESSION, datatype='emg', root=DIR_BIDS_ROOT)
write_raw_bids(
    raw_emg,
    bids_path_emg,
    format='EDF',
    emg_placement='Other',
    allow_preload=True,
    overwrite=True,
    verbose=True,
)

# complete the emg.json sidecar (SENIAM placement description + acquisition metadata)
emg_sidecar = bids_path_emg.copy().update(suffix='emg', extension='.json')
update_sidecar_json(
    bids_path=emg_sidecar,
    entries={
        'Manufacturer': 'Delsys',
        'ManufacturersModelName': 'Trigno',
        'EMGReference': 'bipolar',
        'EMGGround': 'n/a',
        'EMGPlacementSchemeDescription': (
            'Surface electrodes placed according to the SENIAM guidelines '
            '(Surface ElectroMyoGraphy for the Non-Invasive Assessment of Muscles).'
        ),
    },
)

print(f'Finished writing EMG BIDS for participant {SUBJECT_ID} and task {TASK}')


####################################
# Mocap 2 BIDS
####################################
# The Qualisys 6D stream stores one row per detected marker per frame as
# [x, y, z, trajectory_id]. Fast / occluded markers (the feet during treadmill
# walking) are repeatedly dropped and re-acquired under brand-new trajectory
# ids, so the raw ids cannot be used to identify markers directly: here the two
# foot markers fragment into ~1000 short-lived ids while the three slow/static
# markers (treadmill, lower back, sternum) keep one stable id for the whole take.
#
# cluster_markers_tracked() reconstructs the 100 Hz frame grid, pins the stable
# (rigid) markers by id (with ghost rejection), and tracks the fragmented foot
# markers by trajectory-id continuity + a gated spatial matcher, returning one
# clean, continuous track per physical marker. See src/clustering_tracked.py for
# the full rationale and the empirical analysis behind it.
mocap_stream = [s for s in streams if s['info']['type'][0] == '6D'][0]

SRATE_MOCAP = float(mocap_stream["info"]["nominal_srate"][0])
TRACKSYS = "qualisys"
N_MARKERS = 5  # treadmill, lower back, sternum, left foot, right foot

print(f"\nRaw mocap: {mocap_stream['time_series'].shape[0]} rows, "
      f"{len(np.unique(mocap_stream['time_series'][:, 3]))} unique trajectory IDs")

# =========================================================================
# Cluster marker rows into one continuous track per physical marker
# =========================================================================
grid, marker_names, cluster_info = cluster_markers_tracked(
    mocap_stream["time_series"],
    mocap_stream["time_stamps"],
    sr=SRATE_MOCAP,
    n_markers=N_MARKERS,
)
n_frames = grid.shape[0]
n_channels = N_MARKERS * 3

# Report clustering quality (QC)
print("\nMarker clustering result:")
print(f"  stable (rigid) IDs: {cluster_info['stable_ids']}")
for c, name in enumerate(marker_names):
    print(f"  {name:12s}: {cluster_info['coverage'][c]:5.1f}% coverage, "
          f"max frame-to-frame jump {cluster_info['max_jump'][c]:.1f} mm")
print(f"  no track collapse: min foot-foot distance "
      f"{cluster_info.get('min_mobile_distance', float('nan')):.1f} mm")
print(f"  foot identity swaps: {cluster_info['id_cross_assignments']}")
print(f"  trajectory outliers removed (e.g. reflections): "
      f"{cluster_info['outliers_removed_total']} {cluster_info['outliers_removed']}")
print(f"  QC checks: {cluster_info['checks']}")
for w in cluster_info["warnings"]:
    print(f"  ! {w}")
if not cluster_info["all_checks_pass"]:
    print("  ! Some QC checks failed - inspect the diagnostic plot before using.")

# Flatten to (n_frames, n_channels): marker0 x,y,z, marker1 x,y,z, ...
motion_data = grid.reshape(n_frames, n_channels)

# =========================================================================
# Diagnostic plot: per-axis position of every clustered marker over time
# =========================================================================
t_grid = np.arange(n_frames) / SRATE_MOCAP
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for c, name in enumerate(marker_names):
    for ax_idx in range(3):
        axes[ax_idx].plot(t_grid, grid[:, c, ax_idx], label=name, alpha=0.8, lw=0.8)
for ax_idx, label in enumerate(["X (mm)", "Y (mm)", "Z (mm)"]):
    axes[ax_idx].set_ylabel(label)
axes[0].legend(loc="upper right", fontsize=8, ncol=N_MARKERS)
axes[-1].set_xlabel("Time (s)")
axes[0].set_title(f"Clustered marker positions - {SUBJECT_ID} {TASK}")
plt.tight_layout()
plt.show()

# =========================================================================
# Create BIDS Channel objects
# =========================================================================
channels = [
    Channel(
        channel_name=f"{marker_names[i]}_{axis}",
        channel_component=axis,
        channel_type="POS",
        channel_tracked_point=marker_names[i],
        channel_units="mm",
    )
    for i in range(N_MARKERS)
    for axis in ["x", "y", "z"]
]

# =========================================================================
# Create MotionData and export to BIDS
# =========================================================================
motion = MotionData(
    # Required fields
    subject=SUBJECT_ID,
    task_name=TASK,
    tracksys=TRACKSYS,
    sampling_frequency=SRATE_MOCAP,
    tracked_points_count=N_MARKERS,
    # Recommended fields
    manufacturer="Qualisys",
    recording_type="continuous",
    motion_channel_count=n_channels,
    recording_duration=n_frames / SRATE_MOCAP,
    # Optional
    session=SESSION if SESSION else None,
    # Data
    data=motion_data,
    channels=channels,
)

try:
    validate_motion_data(motion)
    print("[OK] Motion data is BIDS compliant!")
except Exception as e:
    print(f"[WARN] Validation warning: {e}")

# Create output directory
motion_dir = create_bids_directory_structure(
    base_dir=DIR_BIDS_ROOT,
    subject=SUBJECT_ID,
    session=SESSION if SESSION else None,
)

# Export
files = export_bids_motion(
    motion,
    out_dir=motion_dir,
    validate=True,
    overwrite=True,
)

# motionbids writes missing samples via numpy.savetxt as the token "nan", but
# BIDS requires "n/a" for missing values. Rewrite the motion .tsv so the output
# is valid BIDS (no float repr contains the substring "nan", so this is safe).
motion_tsv = Path(files["tsv"])
tsv_text = motion_tsv.read_text()
if "nan" in tsv_text:
    motion_tsv.write_text(tsv_text.replace("nan", "n/a"))

print(f"[OK] Created BIDS motion files: {list(files.keys())}")
print(f"  Output directory: {motion_dir}")
