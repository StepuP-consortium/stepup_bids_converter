import json
from mne_bids import make_dataset_description, write_raw_bids, BIDSPath
from mnelab.io.xdf import read_raw_xdf
import mne
import numpy as np
from pathlib import Path
import pyxdf

from utils.motionbids import generate_channels_tsv, generate_motion_json_file
from utils.config import DIR_BIDS_ROOT

# load data
file_path = Path(r"C:\Users\juliu\Desktop\kiel\stepup_bids_converter\data\source\PILOT _OLI_17062025")  # Replace with your XDF file path)

# find all xdf files which include the string walk in the filename in the directory and print them each file name
xdf_files = list(file_path.glob('*Walk*.xdf'))
if not xdf_files:
    print("No XDF files found in the directory.")
else:
    for xdf_file in xdf_files:
        print(f"Found XDF file: {xdf_file.name}")

# load the first xdf file
streams, fileheader = pyxdf.load_xdf(file_path.joinpath(xdf_files[0]), handle_clock_resets=False)
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


subject_id = 'TestBologna'
task = 'FixSpeed'
# load EEG stream into MNE and export to BIDS
eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
eeg_stream_id = eeg_stream['info']['stream_id']

raw = read_raw_xdf(fname=file_path.joinpath(xdf_files[0]), stream_ids=[eeg_stream_id], prefix_markers=True)  # this is a mne.io.Raw object
raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

# delete events if they start before eeg recording
events = mne.events_from_annotations(raw)
bids_path = BIDSPath(subject=subject_id, task=task, datatype='eeg', root=DIR_BIDS_ROOT)
write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='BrainVision', verbose=True)

print(f'Finished writing BIDS for participant {subject_id} and task {task}')


# Convert the EEG stream to MNE Raw object
import mne
raw_eeg = mne.io.RawArray(eeg_stream['time_series'].T, mne.create_info(ch_names=eeg_stream['info']['desc'][0]['channels']['channel'], sfreq=eeg_stream['info']['nominal_srate'][0]))
# Set the first channel as the time channel
raw_eeg.set_channel_types({raw_eeg.ch_names[0]: 'time'})
# Save the EEG data to BIDS format
bids_path_eeg = BIDSPath(subject='TestBologna', session='T1', task='FixSpeed', datatype='eeg', root=DIR_BIDS_ROOT).mkdir()
raw_eeg.save(bids_path_eeg.copy().update(suffix='eeg', extension='.fif'), overwrite=True)

# find Mocap stream

lsl_times = eeg_stream['time_stamps'] - eeg_stream['time_stamps'][0]
data_raw = eeg_stream['time_series']

# print unique marker ids, and number of occurences per makrer id
marker_ids = data_raw[:,3]
unique_marker_ids = set(marker_ids)
for marker_id in unique_marker_ids:
    print(f"Marker {int(marker_id)}: {np.sum(marker_ids == marker_id)}")
    
# remove all markers which have less than 100 frames
for marker_id in unique_marker_ids:
    if np.sum(marker_ids == marker_id) < 150:
        idx = np.where(marker_ids == marker_id)
        data_raw = np.delete(data_raw, idx, axis=0)
        marker_ids = np.delete(marker_ids, idx)
        
# Count occurrences of each marker ID
marker_ids = data_raw[:, 3]
unique_marker_ids, counts = np.unique(marker_ids, return_counts=True)

# Get the indices and values of most common markers
most_common_markers = sorted(zip(unique_marker_ids, counts), key=lambda x: x[1], reverse=True)

# Store the data and indices for the 5 most common markers in a dictionary
marker_data_dict = {}
for marker_id, count in most_common_markers:
    indices = np.where(marker_ids == marker_id)
    marker_data = data_raw[indices]
    marker_data_dict[int(marker_id)] = {
        "data": marker_data,
        "indices": indices
    }

# Print the dictionary for verification
for marker_id, info in marker_data_dict.items():
    print(f"Marker {marker_id}:")
    print(f"  Count: {len(info['data'])}")
    print(f"  Indices: {info['indices']}")
print(f"Number of markers: {len(unique_marker_ids)}")

import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Parameters
k = 15  # Number of markers (adjust based on your data)
window_size = 10  # Number of frames to look back

# Storage for cluster centers
prev_centroids = None

# Process data frame by frame
for frame in sorted(df["frame"].unique()):
    frame_data = df[df["frame"] == frame][["x", "y", "z"]].values
    
    # Initialize with first frame
    if prev_centroids is None:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(frame_data)
        prev_centroids = kmeans.cluster_centers_
    else:
        # Assign markers based on previous centroids
        distances = cdist(frame_data, prev_centroids)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids smoothly (weighted moving average)
        new_centroids = np.zeros_like(prev_centroids)
        for i in range(k):
            assigned_points = frame_data[labels == i]
            if len(assigned_points) > 0:
                new_centroids[i] = 0.8 * prev_centroids[i] + 0.2 * np.mean(assigned_points, axis=0)
            else:
                new_centroids[i] = prev_centroids[i]  # Keep old position if marker disappears

        prev_centroids = new_centroids
    
    # Assign corrected labels back
    df.loc[df["frame"] == frame, "marker_label"] = labels


# create bids dataset
TRACKSYS = 'Qualisys'

# Create a dataset description
make_dataset_description(path=DIR_BIDS_ROOT, name='StepUp')

bids_path = BIDSPath(subject='TestBologna', session='T1', task='FixSpeed', datatype='motion', root=DIR_BIDS_ROOT).mkdir()

# write channels.tsv to path
channels = generate_channels_tsv(["Pelvis", "LeftFoot", "RightFoot"])
channels_tsv = bids_path.copy().update(suffix='channels', extension='.tsv')
channels_tsv = str(channels_tsv).split('_channels')
channels_tsv = channels_tsv[0] + '_tracksys-' + TRACKSYS + '_channels' + channels_tsv[1]
channels.to_csv(channels_tsv, sep='\t', index=False)

# write motion.json to path
motion_fields = dict(TaskName='FixSpeed', SamplingFrequency=100)
motion_json = generate_motion_json_file(motion_fields)
motion_json_file = bids_path.copy().update(suffix='motion', extension='.json')
motion_json_file = str(motion_json_file).split('_motion')
motion_json_file = motion_json_file[0] + '_tracksys-' + TRACKSYS + '_motion' + motion_json_file[1]
with open(Path(motion_json_file), 'w') as f:
    json.dump(motion_json, f)
    
# write raw data as tsv
# pelvis id = 394
# left foot id = 1439
# right foot id = 1448
# make vstack in order pelvis, left foot, right foot
pelvis = dict_raw['394']
left_foot = dict_raw['1439']
right_foot = dict_raw['1448']
raw_data = np.vstack([pelvis, left_foot, right_foot]).T
#remove singleton dimension
raw_data = np.squeeze(raw_data)

#store as tsv without headers
raw_data_tsv = str(bids_path.copy().update(suffix='motion', extension='.tsv'))

# split string before _motion and insert tracksys-Qualisys
raw_data_tsv = raw_data_tsv.split('_motion')
raw_data_tsv = raw_data_tsv[0] + '_tracksys-' + TRACKSYS + '_motion' + raw_data_tsv[1]

np.savetxt(raw_data_tsv, raw_data, delimiter='\t', header='', comments='')