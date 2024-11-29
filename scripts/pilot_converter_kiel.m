% Convert source data to BIDS
clc; clear all; 

%% Get the full path of the current script and set directory names
dir_project= fullfile('C:\Users\juliu\Desktop\kiel\stepup_bids_converter');
dir_sourcedata = fullfile(dir_project, 'data', 'source');
dir_bidsdata = fullfile(dir_project, 'data', 'bids');

% add fieltrip
addpath('C:\Users\juliu\Documents\MATLAB\fieldtrip-20240129')
ft_defaults

[filepath,~,~] = fileparts(which('ft_defaults'));
addpath(fullfile(filepath, 'external', 'xdf'));
addpath(fullfile(dir_project,"utils_matlab"));

%% find all xdf files in sourcedata
xdfFiles = dir(fullfile(dir_sourcedata, '*.xdf'));
xdfFileName = fullfile(xdfFiles.folder,xdfFiles.name);

%% 1. load and inspect xdf
%--------------------------------------------------------------------------
streams                         = load_xdf(xdfFileName);
[streamNames, channelMetaData]  = xdf_inspect(streams); 

% keep track of data modalities (find entries from output)
% indices are better found this way because stream order may differ between
% recordings 

EEGStreamName           = 'BrainVision RDA'; 
EEGStreamInd            = find(strcmp(streamNames, EEGStreamName)); 

EMGStreamName         = 'DelSys'; 
EMGStreamInd          = find(strcmp(streamNames, EMGStreamName)); 

MotionStreamName        = 'Qualisys';
MotionStreamInd         = find(strcmp(streamNames, MotionStreamName)); 

% 2. convert streams to fieldtrip data structs  
%--------------------------------------------------------------------------
EEGftData           = stream_to_ft(streams{EEGStreamInd}); 
EMGftData           = stream_to_ft(streams{EMGStreamInd});
MotionftData        = stream_to_ft(streams{MotionStreamInd}); 

% 3. Save time synch information
%--------------------------------------------------------------------------

% compute difference between onset times
onsetDiffMoCap  = MotionftData.hdr.FirstTimeStamp - EEGftData.hdr.FirstTimeStamp; 
onsetDiffEMG    = EMGftData.hdr.FirstTimeStamp - EEGftData.hdr.FirstTimeStamp;

% time synchronization using scans.tsv acq field
% later to be entered as cfg.scans.acq_time = string, should be formatted according to RFC3339 as '2019-05-22T15:13:38'
eegOnset        = [1990,01,01,00,00,0.000];             % [YYYY,MM,DD,HH,MM,SS]
motionOnset     = [1990,01,01,00,00,onsetDiffMoCap];
emgOnset        = [1990,01,01,00,00,onsetDiffEMG];

eegAcqNum       = datenum(eegOnset);
eegAcqTime      = datestr(eegAcqNum,'yyyy-mm-ddTHH:MM:SS.FFF');
motionAcqNum    = datenum(motionOnset);
motionAcqTime   = datestr(motionAcqNum,'yyyy-mm-ddTHH:MM:SS.FFF');
emgAcqNum       = datenum(emgOnset);
emgAcqTime      = datestr(emgAcqNum,'yyyy-mm-ddTHH:MM:SS.FFF');

%% set cfg for BIDS conversion
cfg = [];

cfg.InstitutionName             = 'Kiel University';

% required for dataset_description.json
cfg.dataset_description.Name                = 'StepuP';
cfg.dataset_description.BIDSVersion         = '1.9';
cfg.method = 'convert'; % the original data is in a BIDS -compliant format and can simply be copied
cfg.bidsroot = dir_bidsdata;  % write to the BIDS directory


% get subject name
subject = strsplit(xdfFiles.name, '_');
subject = [subject{2},subject{3}];
cfg.sub = subject;

% add participant information
cfg.participants.age = 58;
    
% 5. enter eeg metadata and feed to data2bids function
%--------------------------------------------------------------------------
cfg.datatype = 'eeg';
cfg.eeg.Manufacturer                = 'BrainProducts';
cfg.eeg.PowerLineFrequency          = 50; 
cfg.eeg.EEGReference                = 'Cz'; 

% time synch information in scans.tsv file
cfg.scans.acq_time  = eegAcqTime; 

data2bids(cfg, EEGftData);

%% EMG data

% bids relevant emg specs
cfg.emg.EMGReference    = 'bipolar';
cfg.datatype = 'emg';
data2bids(cfg, EMGftData);

%% MoCap data

% prepare mocap data
mocap = prep_omc_bids(MotionftData);
mocap.label = {'PelvisPosX','PelvisPosY','PelvisPosZ',...
                'LeftHeelPosX','LeftHeelPosY','LeftHeelPosZ',...
                'RightHeelPosX','RightHeelPosY','RightHeelPosZ',...
                }';

mocap = ft_datatype_raw(mocap);

% bids relevant mocap
cfg.tracksys                    = 'qualisys';
cfg.motion.TrackingSystemName   = 'qualisys';
cfg.motion.SpatialAxes          = 'BRU';
cfg.motion.samplingrate         = 100;

% specify channel details, this overrides the details in the original data structure
cfg.channels = [];
cfg.channels.name = mocap.label;
cfg.channels.component = {'x' , 'y', 'z', ...
                         'x' , 'y', 'z', ...
                         'x' , 'y', 'z'};
cfg.channels.type = cellstr(repmat('POS',length(mocap.label),1));
cfg.channels.units = cellstr(repmat('mm',length(mocap.label),1));

cfg.channels.tracked_point = {
  'pelvis',
  'pelvis',
  'pelvis'  
  'left_heel',
  'left_heel',
  'left_heel',
  'right_heel',
  'right_heel',
  'right_heel',
  };

cfg.datatype = 'motion';
data2bids(cfg, mocap);


