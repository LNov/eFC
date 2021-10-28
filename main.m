clear
% Load example data as 3D array (n_parcels x n_frames x n_subjects)
% In the manuscript, data is a (200 x 1200 x 100) array of time series
% from the HCP dataset, preprocessed and cleaned via ICA-FIX, and
% parcellated using the 'Schaefer2018_200Parcels_17Networks' parcellation
load('example_data.mat','data');
T     	= size(data,2); % time steps
subj_n	= size(data,3); % number of subjects
N       = 10; % number of parcels (WARNING: very slow with N = 200)
data    = data(1:N,:,:);
data    = data - mean(data,1); % GSR
data 	= zscore(data,0,2);

%% loop over subjects and compute empirical and null edge-centric measures
for i_subj = 1:subj_n
    fprintf('Analysing subject %i...\n',i_subj)
    % Select time series for current subject (i_subj)
    ts                                  = data(:,:,i_subj); 
    % Analyse current subject
    r                               	= efc_single_subject(ts);
    % store results
    res.nCov_eigvals(i_subj,:)        	= r.nCov_eigvals;
    res.RSS(i_subj,:)                  	= r.RSS;
    res.RSS_est_wishart_short(i_subj,:)	= r.RSS_est_wishart_short;
    res.RSS_est_short(i_subj,:)        	= r.RSS_est_short;
    res.pval_ks(i_subj)               	= r.pval_ks;
    res.pval_ks_wishart(i_subj)       	= r.pval_ks_wishart;
    res.FC_sim_top(i_subj,:)          	= r.FC_sim_top;
    res.FC_sim_bot(i_subj,:)           	= r.FC_sim_bot;
    res.FC_sim_top_est(i_subj,:)       	= r.FC_sim_top_est;
    res.FC_sim_bot_est(i_subj,:)     	= r.FC_sim_bot_est;
    res.FC_sim_x_top(i_subj,:)         	= r.FC_sim_x_top;
    res.FC_sim_x_bot(i_subj,:)        	= r.FC_sim_x_bot;
    res.FC_mod_top(i_subj,:)          	= r.FC_mod_top;
    res.FC_mod_bot(i_subj,:)          	= r.FC_mod_bot;
    res.FC_mod_top_est(i_subj,:)       	= r.FC_mod_top_est;
    res.FC_mod_bot_est(i_subj,:)       	= r.FC_mod_bot_est;
    res.eFC_sim(i_subj)               	= r.eFC_sim;
end

%% call plotting script to generate figures
plots