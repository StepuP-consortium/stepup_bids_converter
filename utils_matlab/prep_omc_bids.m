function ft_data_bids = prep_omc_bids(ft_data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

marker_ids = MotionftData.trial{1}(4,:);

idx_pelvis = marker_ids == 394;
idx_lf = marker_ids == 1439;
idx_rf = marker_ids == 1448;

ft_data_bids = ft_data;
ft_data_bids.trial =  {}
ft_data



end