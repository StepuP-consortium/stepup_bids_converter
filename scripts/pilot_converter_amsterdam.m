% Convert source data to BIDS
clc; clear all; 

%% load data

dir_source = fullfile(pwd, 'data', 'source');
dir_bids = fullfile(pwd, 'data', 'bids');

% add fieltrip
addpath('C:\Users\User\Documents\MATLAB\toolboxes\fieldtrip-20230503')
ft_defaults

%% set cfg for BIDS conversion
cfg = [];

cfg.InstitutionName             = 'VU';

% required for dataset_description.json
cfg.dataset_description.Name                = 'StepuP';
cfg.dataset_description.BIDSVersion         = '1.8';
cfg.method = 'convert'; % the original data is in a BIDS -compliant format and can simply be copied
cfg.bidsroot = './data/bids';  % write to the present working directory


%% find all datasets in sourcedata
datasets = dir(fullfile(dir_source, '*.mat'));

%% loop over datasets
for i = 1:length(datasets)
    % load data
    data = load(fullfile(datasets(i).folder, datasets(i).name));
    
    % get subject name
    subject = strsplit(datasets(i).name, '_');
    subject = subject{1};
    
    % bids relevat  modality agnostic specs
    cfg.sub = subject;

    % retrieve EEG data
    eeg = data.data_EEG;

    % bids relevant eeg
    cfg.datatype = 'eeg';
    ft_checkdata(eeg);

    % convert eeg 2 bids
    cfg.eeg.PowerLineFrequency = 50;   % since recorded in the EU
    cfg.eeg.EEGReference       = 'Cz'; % actually I do not know, but let's assume it was left mastoid
    cfg.datatype  = 'eeg';
    data2bids(cfg, eeg);

    % read EMG data to fieldtrip format
    emg.trial{1} = data.EMG;
    emg.labels{1} = data.EMG_labels;
    
    ft_checkdata(emg);

    % bids relevant emg specs
    cfg.emg.EMGReference    = 'bipolar';
    data2bids(cfg, emg);
    
end
