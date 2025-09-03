import json
import matplotlib.pyplot as plt
from mne_bids import make_dataset_description, write_raw_bids, BIDSPath
from mnelab.io.xdf import read_raw_xdf
import mne
import numpy as np
from pathlib import Path
import pyxdf
import os

from utils.motionbids import generate_channels_tsv, generate_motion_json_file
from utils.config import DIR_BIDS_ROOT, DIR_PROJ
from utils.plotting import plot_marker_events
from utils.clustering import merge_marker_fragments

##################################
# Create a dataset description
##################################
# This is a required step for BIDS datasets
# It creates a dataset_description.json file in the BIDS root directory
# make_dataset_description(path=DIR_BIDS_ROOT, name='StepUp_Bologna') # ERROR: name 'make_dataset_description' is not defined


###############################################################################
# EEG 2 BIDS
###############################################################################
# load data



subject_id = "Bol14"
session = "T0"
fname = "test"
task = "WALKING_14_1"

# Costruzione del percorso
file_name = f"{fname}_{task}.xdf"  # o qualsiasi convenzione tu voglia
file_path = os.path.join(
    r"C:\Users\juliu\Desktop\kiel\stepup_bids_converter\data\source",
    subject_id,
    session,
    file_name
)

# find all xdf files which include the string walk in the filename in the directory and print them each file name
# xdf_files = list(file_path.glob('*test*.xdf'))
# if not xdf_files:
#     print("No XDF files found in the directory.")
# else:
#     for xdf_file in xdf_files:
#         print(f"Found XDF file: {xdf_file.name}")

# load the first xdf file
# streams, fileheader = pyxdf.load_xdf(xdf_files[0], handle_clock_resets=False)
streams, header = pyxdf.load_xdf(file_path)
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

# fname = xdf_files[4].name  # 4 = WALKING_14_1; 
# # subject_id -> already defined 
# task = fname.split("_")[1] # Estract task from filename (after "test_")
# session = timepoint  # Already defined 

# load EEG stream into MNE and export to BIDS
eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
eeg_stream_id = eeg_stream['info']['stream_id']

raw = read_raw_xdf(fname=file_path, stream_ids=[eeg_stream_id], prefix_markers=True)  # this is a mne.io.Raw object

# make montage based on the channel names
# montage = mne.channels.make_standard_montage('standard_1005')
# raw.set_montage(montage)

# set info for BIDS
raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

# write raw data to BIDS
bids_task = 'WalkingFS'
bids_path_eeg = BIDSPath(subject=subject_id, task=bids_task, session=session, datatype='eeg', root=DIR_BIDS_ROOT)
write_raw_bids(raw, bids_path_eeg, overwrite=True, allow_preload=True, format='BrainVision', verbose=True)

print(f'Finished writing EEG BIDS for participant {subject_id} and task {task}')

###############################################################################
# EMG 2 BIDS
###############################################################################
# find EMG stream
emg_stream = [s for s in streams if s['info']['type'][0] == 'EMG'][0]
emg_stream_id = emg_stream['info']['stream_id']

sensor_names = [
    'RfEmgR', 'RfEmgL','BfEmgR', 'BfEmgL', 
    'GaEmgR', 'GaEmgL', 'GmEmgR', 'GmEmgL'
]

emg_data = np.array(emg_stream['time_series'][:,:8])
emg_dict = {}
for i, name in enumerate(sensor_names):
    emg_dict[name] = emg_data[:, i]   # prendo direttamente la colonna i

# Merge all sensor data into a 2D array (samples x sensors)
emg_2d = np.column_stack([emg_dict[name] for name in sensor_names])

# Plot



# Supponendo che emg_2d sia una matrice Nx8 e sensor_names sia una lista di 8 nomi
plt.figure(figsize=(20, 20))  # figura alta per 8 subplot

for i, sensor_name in enumerate(sensor_names):
    plt.subplot(8, 1, i+1)  # 8 subplot verticali
    plt.plot(emg_2d[:, i])
    plt.title(sensor_name, fontsize=14, fontweight='bold', fontstyle='italic')
    plt.xlim([0, 50*2000])  # se vuoi limitare a 0-60 secondi
    plt.ylim([-500,500])
    if i == 7:
        plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('EMG', fontsize=12)

plt.suptitle('EMG Signals', fontsize=24, fontweight='bold', y=0.95)
plt.subplots_adjust(hspace=0.6)  # spazio tra i subplot
plt.show()

# write data to BIDS
bids_path = BIDSPath(subject=subject_id, task=bids_task, session=session, datatype='emg', root=DIR_BIDS_ROOT)
bids_path.mkdir()

# write emg.tsv and channels.tsv for emg


###############################################################################
# Mocap 2 BIDS
###############################################################################
# create bids path for motion capture data
#bids_path_motion = BIDSPath(subject=subject_id, task=task, session=session, datatype='motion', root=DIR_BIDS_ROOT) #ERROR

# find Mocap stream
mocap_stream = [s for s in streams if s['info']['type'][0] == 'MoCap'][0]

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

dict_raw = {}
fig, ax = plt.subplots(3, 1, figsize=(10, 8)) 
for marker_id in unique_marker_ids:
    idx = np.where(marker_ids == marker_id)
    
    # Aggiungi label con il marker_id
    ax[0].plot(mocap_times[idx], mocap_raw[idx, 0].T, label=f'ID {int(marker_id)}')
    ax[1].plot(mocap_times[idx], mocap_raw[idx, 1].T, label=f'ID {int(marker_id)}')
    ax[2].plot(mocap_times[idx], mocap_raw[idx, 2].T, label=f'ID {int(marker_id)}')
    
    dict_raw[str(int(marker_id))] = mocap_raw[idx, :].T

# Aggiungi legende ai subplot
for a in ax:
    a.legend()
    a.set_xlabel('Time [s]')
    a.set_ylabel('Position')

plt.tight_layout()
plt.show()

# Example usage:
merged_markers = merge_marker_fragments(marker_data_dict, n_expected_markers=5)

# create bids dataset
TRACKSYS = 'Vicon'  # Track system used for motion capture
SRATE_MOCAP = float(mocap_stream['info']['nominal_srate'][0])  # Sampling frequency for motion capture data


# plot marker data
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for marker_id, info in marker_data_dict.items():
    ax.plot(info['data'][:, 0], info['data'][:, 2], label=f'Marker {marker_id}')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')


# FILLING THE GAPS 

data = mocap_raw
time = mocap_times

threshold_dist=55.0
threshold_time=0.5

unique_ids = np.unique(data[:, 3])  # Otteniamo gli ID unici dei marker

merged_markers = {}

for marker_id in unique_ids:
    
    min_len = min(data.shape[0], len(time))
    if data.shape[0] != len(time):
        data = data[:min_len, :]
        time = time[:min_len]
    
    # Seleziona solo i punti con l'ID attuale
    mask = data[:, 3] == marker_id
    marker_data = data[mask, :3]   # Solo coordinate x, y, z
    marker_time = time[mask]       # Tempi corrispondenti
    
    found_match = False
    for existing_id in merged_markers.keys():
        #print(merged_markers)
        
        existing_data, existing_time = merged_markers[existing_id]
         
        dist = np.linalg.norm(existing_data[-1] - marker_data[0])  
        min_dist = np.min(dist)
        time_gap = abs(existing_time[-1] - marker_time[0])
        min_time = np.min(time_gap)
        #time_gap = abs(existing_time[-1] - marker_time[0])  # Differenza temporale
        
        if min_dist < threshold_dist and min_time < threshold_time:
        #if dist < threshold_dist and time_gap < threshold_time:
            nan_row = np.full((1, 3), np.nan)
            nan_time = np.array([np.nan])
                
            existing_data = np.vstack((existing_data, nan_row))
            existing_time = np.hstack((existing_time, nan_time))

            # Unisce i segmenti con eventuali NaN
            merged_markers[existing_id] = (
                np.vstack((existing_data, marker_data)),
                np.hstack((existing_time, marker_time))
            )
            found_match = True
            print(f'match: {found_match} existing_id: {existing_id} marker_id: {marker_id}')

            break
        
    if not found_match:
        merged_markers[marker_id] = (marker_data, marker_time)


final_markers = []
final_times = []
final_ids = list(merged_markers.keys())[:16]  

for marker_id in final_ids:
    marker_data, marker_time = merged_markers[marker_id]
    marker_id_col = np.full((marker_data.shape[0], 1), marker_id)  
    final_markers.append(np.hstack((marker_data, marker_id_col)))
    final_times.append(marker_time)


data_merged = np.vstack(final_markers)
time_merged = np.hstack(final_times)


time_merged = time_merged - time_merged[0]

indx_merged = np.array(sorted(set(data_merged[:, 3])))

data_dict_merged = {}  
time_dict_merged = {}  
for i in range(0,np.shape(indx_merged)[0]):
    mask = data_merged[:, 3] == int(indx_merged[i])
    marker = data_merged[mask, 0:3]
    t = time_merged[mask]
    
    data_dict_merged[f"marker_{i+1}"] = {"marker": marker}
    time_dict_merged[f"time_{i+1}"] = {"time": t}

plt.figure(figsize=(18, 9))

for comp in range(3):
    plt.subplot(3, 1, comp+1)
    
    for i in range(len(indx_merged)):
        t = time_dict_merged[f"time_{i+1}"]["time"]
        m = data_dict_merged[f"marker_{i+1}"]["marker"]
        plt.plot(t, m[:, comp], label=f'ID {int(indx_merged[i])}')
        
    plt.xlim([0, max(time_merged)])
    plt.title(f'Component {comp+1}', fontsize=24, fontweight='bold')
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    
    # Legenda a destra
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()


# write raw data as tsv
# make vstack in order pelvis, left foot, right foot
# ES Bol14 - T0 
# -> es. walking_14_1 pelvis 51; left 57; right 44
# -> es. walking_cs_1 pelvis 60; left 58; right 59
# -> es. fam_2 pelvis 18; left 1; right 16
id_pelvis = 51;
id_left_foot = 57;
id_right_foot = 44;

pelvis = mocap_raw[mocap_raw[:, 3] == id_pelvis, :]
left_foot = mocap_raw[mocap_raw[:, 3] == id_left_foot, :]
right_foot = mocap_raw[mocap_raw[:, 3] == id_right_foot, :]

raw_data = np.vstack([pelvis, left_foot, right_foot])
#remove singleton dimension
raw_data = np.squeeze(raw_data)

## TO CHECK 

#store as tsv without headers
raw_data_tsv = str(bids_path.update(suffix='motion', extension='.tsv'))

# split string before _motion and insert tracksys-Qualisys
raw_data_tsv = raw_data_tsv.split('_motion')
raw_data_tsv = raw_data_tsv[0] + '_tracksys-' + TRACKSYS + '_motion' + raw_data_tsv[1]

np.savetxt(raw_data_tsv, raw_data, delimiter='\t', header='', comments='')

# write channels.tsv to path
channels = generate_channels_tsv(["Pelvis", "LeftFoot", "RightFoot"])
channels_tsv = bids_path.update(suffix='channels', extension='.tsv')
channels_tsv = str(channels_tsv).split('_channels')
channels_tsv = channels_tsv[0] + '_tracksys-' + TRACKSYS + '_channels' + channels_tsv[1]
channels.to_csv(channels_tsv, sep='\t', index=False)

# write motion.json to path
motion_fields = dict(TaskName=task, SamplingFrequency=SRATE_MOCAP)
motion_json = generate_motion_json_file(motion_fields)
motion_json_file = bids_path.update(suffix='motion', extension='.json')
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
