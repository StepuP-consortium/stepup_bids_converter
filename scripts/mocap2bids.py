import json
from mne_bids import make_dataset_description, BIDSPath
import numpy as np
from pathlib import Path
import pyxdf

from utils.motionbids import generate_channels_tsv, generate_motion_json_file
from utils.config import DIR_BIDS_ROOT

# load data
file_path = r"C:\Users\juliu\Desktop\kiel\stepup_setup_jw\data\Test_bologna_25_03_25\4_WALKING_14\sub-P001_ses-S001_task-Default_run-001_eeg_old6.xdf"  # Replace with your XDF file path

streams, fileheader = pyxdf.load_xdf(file_path, handle_clock_resets=False)
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


# find Mocap stream
mocap_stream = [s for s in streams if s['info']['type'][0] == 'MoCap'][0]

mocap_times = mocap_stream['time_stamps'] - mocap_stream['time_stamps'][0]
mocap_raw = mocap_stream['time_series']

# print unique marker ids, and number of occurences per makrer id
marker_ids = mocap_raw[:,3]
unique_marker_ids = set(marker_ids)
for marker_id in unique_marker_ids:
    print(f"Marker {int(marker_id)}: {np.sum(marker_ids == marker_id)}")
    
# remove all markers which have less than 100 frames
for marker_id in unique_marker_ids:
    if np.sum(marker_ids == marker_id) < 150:
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