import json
import matplotlib.pyplot as plt
from mne_bids import make_dataset_description, write_raw_bids, update_sidecar_json, BIDSPath
from mnelab.io.xdf import read_raw_xdf
import mne
import numpy as np
from pathlib import Path
import pyxdf

from src.motionbids import generate_channels_tsv, generate_motion_json_file
from src.config import DIR_BIDS_ROOT, DIR_PROJ
from src.plotting import plot_marker_events
from src.clustering import merge_marker_fragments
from src.xdf import get_effective_srate_xdf

##################################
# Create a dataset description
##################################
# This is a required step for BIDS datasets
# It creates a dataset_description.json file in the BIDS root directory
make_dataset_description(path=DIR_BIDS_ROOT, name='StepUp_Kiel')




###################################
# EEG 2 BIDS
###################################
# load data
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
subject_id = fname.split('_')[2]  # Extract subject ID from filename
task = fname.split('_')[-1].split('.')[0]  # Extract task from filename
session = fname.split('_')[3]  # Extract visit from filename

# load EEG stream into MNE and export to BIDS
eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
eeg_stream_id = eeg_stream['info']['stream_id']

raw = read_raw_xdf(fname=file_path.joinpath(xdf_files[0]), stream_ids=[eeg_stream_id], prefix_markers=True)  # this is a mne.io.Raw object

# set montage using the provided .bvef file for channel locations
bvef_path = DIR_PROJ.joinpath('utils', 'montages', 'stepup_kiel.bvef')
montage = mne.channels.read_custom_montage(str(bvef_path))
raw.set_montage(montage)

# set info for BIDS
raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

# write raw data to BIDS
bids_path_eeg = BIDSPath(subject=subject_id, task=task, session=session, datatype='eeg', root=DIR_BIDS_ROOT)
write_raw_bids(raw, bids_path_eeg, overwrite=True, allow_preload=True, format='BrainVision', verbose=True)

print(f'Finished writing EEG BIDS for participant {subject_id} and task {task}')




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
bids_path_emg = BIDSPath(subject=subject_id, task=task, session=session, datatype='emg', root=DIR_BIDS_ROOT)
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

print(f'Finished writing EMG BIDS for participant {subject_id} and task {task}')











####################################
# Mocap 2 BIDS
####################################
# create bids path for motion capture data
bids_path_motion = BIDSPath(subject=subject_id, task=task, session=session, datatype='motion', root=DIR_BIDS_ROOT)

# find Mocap stream
mocap_stream = [s for s in streams if s['info']['type'][0] == '6D'][0]

mocap_times = mocap_stream['time_stamps'] - mocap_stream['time_stamps'][0]
mocap_raw = mocap_stream['time_series']


# print unique marker ids, and number of occurences per makrer id
marker_ids = mocap_raw[:,3]
unique_marker_ids = set(marker_ids)
for marker_id in unique_marker_ids:
    print(f"Marker {int(marker_id)}: {np.sum(marker_ids == marker_id)}")
    
# remove all markers which have less than 30 frames
for marker_id in unique_marker_ids:
    if np.sum(marker_ids == marker_id) < 30:
        idx = np.where(marker_ids == marker_id)
        mocap_raw = np.delete(mocap_raw, idx, axis=0)
        marker_ids = np.delete(marker_ids, idx)

     
# Count occurrences of each marker ID
marker_ids = mocap_raw[:, 3]
unique_marker_ids, counts = np.unique(marker_ids, return_counts=True)

# Get the indices and values of most common markers
most_common_markers = sorted(zip(unique_marker_ids, counts), key=lambda x: x[1], reverse=True)

# Store the data and indices for the 5 most common markers in a dictionary
marker_data_dict = {}
for marker_id, count in most_common_markers:
    indices = np.where(marker_ids == marker_id)
    marker_data = mocap_raw[indices]
    marker_data_dict[int(marker_id)] = {
        "data": marker_data,
        "indices": indices
    }


# Print the dictionary for verification
for marker_id, info in marker_data_dict.items():
    indices = info['indices']
    print(f"Marker {marker_id}:")
    if len(indices) > 0:
        print(f"  First index: {indices[0][0]}, Last index: {indices[0][-1]}")
    else:
        print("  No indices found.")


# plot marker events
fig = plot_marker_events(marker_data_dict, show=False)


# plot ssecond against time for each unique marker id
dict_raw = {}
fig, ax = plt.subplots(3, 1)
for marker_id in unique_marker_ids:
    idx = np.where(marker_ids == marker_id)
    ax[0].plot(mocap_times[idx], mocap_raw[idx, 0].T)
    ax[1].plot(mocap_times[idx], mocap_raw[idx, 1].T)
    ax[2].plot(mocap_times[idx], mocap_raw[idx, 2].T)
    dict_raw[str(int(marker_id))] = mocap_raw[idx,:].T
#ax.set_xlim([0, 10])
#ax.set_ylim([393.8, 394.2])

# Example usage:
merged_markers = merge_marker_fragments(marker_data_dict, n_expected_markers=5)

# create bids dataset
TRACKSYS = 'Qualisys'  # Track system used for motion capture
SRATE_MOCAP = float(mocap_stream['info']['nominal_srate'][0])  # Sampling frequency for motion capture data


# plot marker data
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for marker_id, info in marker_data_dict.items():
    ax.plot(info['data'][:, 0], info['data'][:, 2], label=f'Marker {marker_id}')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

# write raw data as tsv
# make vstack in order pelvis, left foot, right foot
pelvis = vicon_data
left_foot = vicon_data
right_foot = vicon_data

raw_data = np.vstack([pelvis, left_foot, right_foot]).T
#remove singleton dimension
raw_data = np.squeeze(raw_data)

#store as tsv without headers
raw_data_tsv = str(bids_path_motion.copy().update(suffix='motion', extension='.tsv'))

# split string before _motion and insert tracksys-Qualisys
raw_data_tsv = raw_data_tsv.split('_motion')
raw_data_tsv = raw_data_tsv[0] + '_tracksys-' + TRACKSYS + '_motion' + raw_data_tsv[1]

np.savetxt(raw_data_tsv, raw_data, delimiter='\t', header='', comments='')

# write channels.tsv to path
channels = generate_channels_tsv(["Pelvis", "LeftFoot", "RightFoot"])
channels_tsv = bids_path_motion.copy().update(suffix='channels', extension='.tsv')
channels_tsv = str(channels_tsv).split('_channels')
channels_tsv = channels_tsv[0] + '_tracksys-' + TRACKSYS + '_channels' + channels_tsv[1]
channels.to_csv(channels_tsv, sep='\t', index=False)

# write motion.json to path
motion_fields = dict(TaskName=task, SamplingFrequency=SRATE_MOCAP)
motion_json = generate_motion_json_file(motion_fields)
motion_json_file = bids_path_motion.copy().update(suffix='motion', extension='.json')
motion_json_file = str(motion_json_file).split('_motion')
motion_json_file = motion_json_file[0] + '_tracksys-' + TRACKSYS + '_motion' + motion_json_file[1]
with open(Path(motion_json_file), 'w') as f:
    json.dump(motion_json, f)



from tslearn.generators import random_walks
X = random_walks(n_ts=50, sz=32, d=1)

# Apply K-Means clustering using DTW as the distance measure
model = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=1)
clusters = model.fit_predict(aligned_series)


# Output the cluster assignments
print(clusters)
