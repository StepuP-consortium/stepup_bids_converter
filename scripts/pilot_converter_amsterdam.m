% Convert source data to BIDS
clc; clear all; 

%% load data

dir_source = fullfile(pwd, 'data', 'source');
dir_bids = fullfile(pwd, 'data', 'bids');

%% find all datasets in sourcedata
datasets = dir(fullfile(dir_source, '*.mat'));


%% loop over datasets
for i = 1:length(datasets)
    % load data
    data = load(fullfile(datasets(i).folder, datasets(i).name));
    
    % get subject name
    subject = strsplit(datasets(i).name, '.');
    subject = subject{1};
    
