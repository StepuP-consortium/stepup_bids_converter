function ft_data_bids = prep_omc_bids(ft_data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% define local srate for MoCap
fs_mocap = 100;

% Identify avaliable marker ids
marker_ids = ft_data.trial{1}(4,:);

% find indices for existing markers
idx_pelvis = marker_ids == 394;
idx_lf = marker_ids == 1439;
idx_rf = marker_ids == 1448;

% prelocated output for speedup
ft_data_bids = ft_data;
ft_data_bids.trial =  {}
data_pelvis = ft_data.trial{1}(1:3, idx_pelvis);
data_lf = ft_data.trial{1}(1:3, idx_lf);
data_rf = ft_data.trial{1}(1:3, idx_rf);

time_vec = linspace(0,length(data_rf) / fs_mocap, length(data_rf));

ft_data_bids.trial{1}   = double([data_pelvis; data_lf; data_rf]);
ft_data_bids.time{1}    = time_vec;

% remove hdr field
ft_data_bids            = rmfield(ft_data_bids, 'hdr');

end