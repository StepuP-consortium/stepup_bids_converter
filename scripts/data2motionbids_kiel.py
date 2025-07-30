import json
import matplotlib.pyplot as plt
from mne_bids import make_dataset_description, write_raw_bids, BIDSPath
from mnelab.io.xdf import read_raw_xdf
import mne
import numpy as np
from pathlib import Path
import pyxdf

from utils.motionbids import generate_channels_tsv, generate_motion_json_file
from utils.config import DIR_BIDS_ROOT, DIR_PROJ
from utils.plotting import plot_marker_events
from utils.clustering import merge_marker_fragments



# load data
file_path = Path(r"C:\Users\juliu\Desktop\kiel\stepup_bids_converter\data\source")  # Replace with your XDF file path)

# find all xdf files which include the string walk in the filename in the directory and print them each file name
xdf_files = list(file_path.glob('*DEU*.xdf'))
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


# Example usage:
merged_markers = merge_marker_fragments(
    marker_data_dict,
    n_expected_markers=5,
    spatial_weight=5.0,       # Higher = more emphasis on spatial similarity
    temporal_weight=0.01,      # Lower = less emphasis on temporal continuity
)

fig = plot_marker_events(merged_markers, show=False)

dict_raw = {}
fig, ax = plt.subplots(3, 1, figsize=(5,15))
for marker_id, info in marker_data_dict.items():
    marker_data = info["data"]
    times = marker_data[:, 0]  # Assuming first column is time
    ax[0].plot(marker_data[:, 0].T, marker_data[:, 1].T)
    ax[0].set_xlabel('X Position')
    ax[0].set_ylabel('Y Position')
    ax[1].plot(marker_data[:, 0].T, marker_data[:, 2].T)
    ax[1].set_xlabel('X Position')
    ax[1].set_ylabel('Z Position')
    ax[2].plot(marker_data[:, 1].T, marker_data[:, 2].T)
    ax[2].set_xlabel('Y Position')
    ax[2].set_ylabel('Z Position')
    dict_raw[str(int(marker_id))] = marker_data.T

# create bids dataset
TRACKSYS = 'Qualisys'  # Track system used for motion capture
SRATE_MOCAP = float(mocap_stream['info']['nominal_srate'][0])  # Sampling frequency for motion capture data

