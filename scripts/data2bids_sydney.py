import json
from mne_bids import make_dataset_description, write_raw_bids, BIDSPath
from mnelab.io.xdf import read_raw_xdf
import mne
import numpy as np
from pathlib import Path
import pandas as pd
import pyxdf

from utils.motionbids import generate_channels_tsv, generate_motion_json_file
from utils.config import DIR_BIDS_ROOT, DIR_PROJ

# load data
file_path = Path(DIR_PROJ.joinpath('data\source\PILOT _OLI_17062025'))  # Replace with your XDF file path)

# find all xdf files which include the string walk in the filename in the directory and print them each file name
xdf_files = list(file_path.glob('*Walk*.xdf'))
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
subject_id = fname.split('_')[0]  # Extract subject ID from filename
task = fname.split('_')[2]  # Extract task from filename
session = fname.split('_')[1]  # Extract visit from filename

############################################
#
#                   EEG
#
############################################
# load EEG stream into MNE and export to BIDS
eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]
eeg_stream_id = eeg_stream['info']['stream_id']

raw = read_raw_xdf(fname=file_path.joinpath(xdf_files[0]), stream_ids=[eeg_stream_id], prefix_markers=True)  # this is a mne.io.Raw object

# make montage based on the channel names
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# set info for BIDS
raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

# write raw data to BIDS
bids_path = BIDSPath(subject=subject_id, task=task, session=session, datatype='eeg', root=DIR_BIDS_ROOT)
write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='BrainVision', verbose=True)

print(f'Finished writing EEG BIDS for participant {subject_id} and task {task}')


############################################
#
#                   MoCap
#
############################################
# find Mocap stream
# load EEG stream into MNE and export to BIDS
data_stream = [s for s in streams if s['info']['type'][0] == 'MoCap'][0]
data_stream_id = data_stream['info']['stream_id']

data_lsl_times = data_stream['time_stamps'] - data_stream['time_stamps'][0]
data_raw = data_stream['time_series']

# <Column A>Timestamps
# <Colum B(0)>FrameNumbers
# <Colum (1)C>Frame rate
# <Colum D(2)>Total Time that the loop has taken for capturing the data from Vicon(It's just for our understanding, don't worry about it)
# <Colum E(3)-CF(82)>Vicon Data<Colum 
# BM(63)-BT(70)>Left Foot and Right Foot(Extra Markers)
# <Colum BU(71)-CF(82)>Unlabeled Markers>
# <Colum CG(83)-AAV(722)EMG Data>We used only two sensors, that 'why all other columns are zero
# <Colum AAW(723)-AHV(904)>Force Plates Data

# extract mocap marker data
mocap_times = data_raw[:,0]
frames_mocap = data_raw[:,1]
vicon_data = data_raw[:,3:83]
emg_data = data_raw[:,82:723]


# create bids dataset
TRACKSYS = 'Vicon'
SRATE_MOCAP = float(data_stream['info']['nominal_srate'][0])  # Sampling frequency for motion capture data

# Create a dataset description
make_dataset_description(path=DIR_BIDS_ROOT, name='StepUp_Sydney')

bids_path = BIDSPath(subject=subject_id, task=task, session=session, datatype='eeg', root=DIR_BIDS_ROOT)
bids_path.datatype = 'motion'
bids_path.mkdir()

# write channels.tsv to path
motion_channels = generate_channels_tsv(["Pelvis", "LeftFoot", "RightFoot"])
motion_channels_tsv = bids_path.copy().update(suffix='channels', extension='.tsv')
motion_channels_tsv = str(motion_channels_tsv).split('_channels')
motion_channels_tsv = motion_channels_tsv[0] + '_tracksys-' + TRACKSYS + '_channels' + motion_channels_tsv[1]
motion_channels.to_csv(motion_channels_tsv, sep='\t', index=False)

# write motion.json to path
motion_fields = dict(TaskName=task, SamplingFrequency=SRATE_MOCAP)
motion_json = generate_motion_json_file(motion_fields)
motion_json_file = bids_path.copy().update(suffix='motion', extension='.json')
motion_json_file = str(motion_json_file).split('_motion')
motion_json_file = motion_json_file[0] + '_tracksys-' + TRACKSYS + '_motion' + motion_json_file[1]
with open(Path(motion_json_file), 'w') as f:
    json.dump(motion_json, f)
    
# write raw data as tsv
# make vstack in order pelvis, left foot, right foot
pelvis = vicon_data[:, 0:3]  # Assuming pelvis markers are in columns 0-5
left_foot = vicon_data[:, 60:63]  # Assuming left foot markers are in columns 60-63
right_foot = vicon_data[:, 64:67]  # Assuming right foot markers are in columns 64-67

motion_data = np.hstack([pelvis, left_foot, right_foot]).squeeze()

#store as tsv without headers
raw_data_tsv = str(bids_path.copy().update(suffix='motion', extension='.tsv'))

# split string before _motion and insert tracksys-Qualisys
raw_data_tsv = raw_data_tsv.split('_motion')
raw_data_tsv = raw_data_tsv[0] + '_tracksys-' + TRACKSYS + '_motion' + raw_data_tsv[1]

np.savetxt(raw_data_tsv, motion_data, delimiter='\t', header='', comments='')

############################################
#
#                   EMG
#
############################################
# find EMG stream and dynamically extract sensor data
emg_sensor_names = [
    'BfEmgR', 'BfEmgL', 'RfEmgR', 'RfEmgL',
    'GaEmgR', 'GaEmgL', 'GmEmgR', 'GmEmgL',
    'TaEmgR', 'TaEmgL'
]

# Define the start and end indices for each sensor (update as needed)
sensor_indices = {
    'BfEmgR': (22, 41),
    'BfEmgL': (43, 62),
    'RfEmgR': (64, 83),
    'RfEmgL': (85, 104),
    'GaEmgR': (106, 125),
    'GaEmgL': (127, 146),
    'GmEmgR': (148, 167),
    'GmEmgL': (169, 188),
    'TaEmgR': (190, 209),
    'TaEmgL': (211, 230),
}

emg_dict = {}
for name in emg_sensor_names:
    start, end = sensor_indices[name]
    emg_dict[name] = emg_data[:, start:end].ravel(order='F')

# Merge all sensor data into a 2D array (samples x sensors)
emg_data_bids = np.column_stack([emg_dict[name] for name in emg_sensor_names])

# create channels.tsv for emg
emg_channels = pd.DataFrame(dict(name=emg_sensor_names, type=['EMG']*len(emg_sensor_names), units=['V']*len(emg_sensor_names)))

# write data to BIDS
bids_path = BIDSPath(subject=subject_id, task=task, session=session, datatype='eeg', root=DIR_BIDS_ROOT)
bids_path.datatype = 'emg'
bids_path.mkdir()

# write emg as tsv
emg_tsv = str(bids_path.copy().update(suffix='emg', extension='.tsv'))
emg_tsv = emg_tsv.split('_emg')
emg_tsv = emg_tsv[0] + '_emg' + emg_tsv[1]
np.savetxt(emg_tsv, emg_data_bids, delimiter='\t', header='', comments='')

# write emg_channels.tsv
channels_tsv = bids_path.copy().update(suffix='channels', extension='.tsv')
channels_tsv = str(channels_tsv).split('_channels')
channels_tsv = channels_tsv[0] + '_channels' + channels_tsv[1]
emg_channels.to_csv(channels_tsv, sep='\t', index=False)
