% Convert source data to BIDS
clc; clear all; 

%% load data
dir_project= fullfile('C:\Users\juliu\Desktop\kiel\stepup_bids_converter');
dir_sourcedata = fullfile('C:\Users\juliu\Desktop\kiel\stepup_bids_converter\data\source');
dir_bidsdata = fullfile(dir_project, 'data', 'bids');

% add fieltrip
addpath('C:\Users\juliu\Desktop\fieldtrip')
ft_defaults

%% set cfg for BIDS conversion
cfg = [];

cfg.InstitutionName             = 'VU';

% required for dataset_description.json
cfg.dataset_description.Name                = 'StepuP';
cfg.dataset_description.BIDSVersion         = '1.9';
cfg.method = 'convert'; % the original data is in a BIDS -compliant format and can simply be copied
cfg.bidsroot = dir_bidsdata;  % write to the present working directory

% time synch information in scans.tsv file

eegAcqNum       = datenum([1990,01,01,00,00,0.000]);
eegAcqTime      = datestr(eegAcqNum,'yyyy-mm-ddTHH:MM:SS.FFF');
cfg.scans.acq_time  = eegAcqTime;

%% find all datasets in sourcedata
datasets = dir(fullfile(dir_sourcedata, '*.mat'));

%% loop over datasets
for i = 1:length(datasets)
    % load data
    data = load(fullfile(datasets(i).folder, datasets(i).name));
    
    % get subject name
    subject = strsplit(datasets(i).name, '_');
    subject = subject{1};
    
    %% EEG data
    % bids relevant  modality agnostic specs
    cfg.sub = ['AMS' subject];
    cfg.task = 'treadmill'

    % retrieve EEG data
    eeg = data.data_EEG;

    % bids relevant eeg
    cfg.datatype = 'eeg';
    ft_checkdata(eeg);

    % convert eeg 2 bids
    cfg.eeg.PowerLineFrequency = 50;   % since recorded in the EU
    cfg.eeg.EEGReference       = 'CAR'; % actually I do not know, but let's assume
    eeg.cfg.event = [];
    data2bids(cfg, eeg);

    %% EMG data
    % read EMG data to fieldtrip format
    emg.trial{1} = data.EMG';
    emg.label = data.EMG_labels';
    emg.time{1}  = linspace(0, length(data.EMG) / 2148, length(data.EMG))
    ft_checkdata(emg);

    % bids relevant emg specs
    cfg.emg.EMGReference    = 'bipolar';
    cfg.datatype = 'emg';
    data2bids(cfg, emg);

    %% MoCap data

    mocap.trial{1} = [data.L_heel, data.R_heel, data.pelvis]';
    mocap.label = {'LeftHeelPosX','LeftHeelPosY','LeftHeelPosZ',...
                    'RightHeelPosX','RightHeelPosY','RightHeelPosZ',...
                    'PelvisPosX','PelvisPosY','PelvisPosZ',...
                    }';
    mocap.time{1} = data.t_qls;

    mocap = ft_datatype_raw(mocap);

    % bids relevant mocap
    cfg.tracksys                    = 'qualisys';
    cfg.motion.TrackingSystemName   = 'qualisys';
    cfg.motion.SpatialAxes          = 'RFU';
    cfg.motion.samplingrate         = data.fs_qls;

    % specify channel details, this overrides the details in the original data structure
    cfg.channels = [];
    cfg.channels.name = mocap.label;
    cfg.channels.component = {'x' , 'y', 'z', ...
                             'x' , 'y', 'z', ...
                             'x' , 'y', 'z'};
    cfg.channels.type = cellstr(repmat('POS',length(mocap.label),1));
    cfg.channels.units = cellstr(repmat('m',length(mocap.label),1))
    
    cfg.channels.tracked_point = {
      'left_heel',
      'left_heel',
      'left_heel',
      'right_heel',
      'right_heel',
      'right_heel',
      'pelvis',
      'pelvis',
      'pelvis'
      };

    cfg.datatype = 'motion';
    data2bids(cfg, mocap);


end
