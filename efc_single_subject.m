function efc_single_subject(i_subj)
% Compute edge-centric features from empirical data (Human Connectome
% Project) and corresponding null model. Compare results in terms of nFC
% similarity from top vs bottom RSS frames, as well as similarity between
% empirical and predicted eFC matrices. Run k-means clustering on eFC.

N           = 200;      % number of parcels
T           = 1200;     % time steps
T_null      = T * 50;	% time steps for null model simulation

% Load HCP ICA-FIX time series for single subject (i_subj)
% using 'Schaefer2018_200Parcels_17Networks' parcellation
ts = data(1:N,1:T,i_subj); % data is a (200 x 1200 x 100) array

% Apply GSR
ts = ts - mean(ts,1);

% z-score
ts = zscore(ts,0,2); % nCov and nFC will coincide in this case

% Match (Faskowitz 2020) ROI order for Schaefer2018_200Parcels_17Networks
load('ROI_order.mat','idx_sort');
ts(51:56,:)     = ts(56:-1:51,:);   % move Limbic_B after Limbic_A
ts(157:164,:)   = ts(164:-1:157,:); % move Limbic_B after Limbic_A
ts              = ts(idx_sort,:);   % sort to match Faskowitz 2020

% Node covariance matrix and node FC
nCov                = (ts * ts') / (T - 1);
nFC                 = corrcov(nCov);

% nCov eigenvalues
eig_nCov            = eig(nCov);
res.nCov_eigvals    = eig_nCov;

% Edge time series
[u,v]               = find(ones(N));
edge_ts_all         = ts(u,:).*ts(v,:);
upper_tri           = logical(triu(ones(N),1)); upper_tri=upper_tri(:);
edge_ts             = edge_ts_all(upper_tri,:);

% Sum of absolute values for each scan
RSS                 = sqrt(sum(edge_ts.^2)); % root sum of squares
res.RSS             = RSS;

% sample Wishart distribution
nCov_chol           = cholcov(nCov); % pass it to wishrnd for speed
ets_est_short       = NaN(N^2,T);
RSS_est_wishart     = NaN(1,T_null);
for t = 1:T_null
    w                       = wishrnd(nCov,1,nCov_chol);
    if t <= T
        ets_est_short(:,t)  = w(:);
    end
    w                       = w(upper_tri);
    RSS_est_wishart(t)      = sqrt(sum(w(:).^2));
end

% RSS from Wishart dist
RSS_est_wishart_short       = RSS_est_wishart(1:T);
res.RSS_est_wishart_short   = RSS_est_wishart_short;

% KS test
[~,pval_ks]                 = kstest2(RSS,RSS_est_wishart);
res.pval_ks_wishart         = pval_ks;

% Null distribution (sum of Gamma RVs)
RSS_est         	= (1/sqrt(2)) * eig_nCov' * chi2rnd(1,N,T_null);
res.RSS_est         = RSS_est;
res.RSS_est_short 	= RSS_est(1:T);

% KS test
[~,pval_ks]         = kstest2(RSS,RSS_est);
res.pval_ks         = pval_ks;

%---------------------------------------------------------------------- 
% get FC estimate from top RSS values
% sort and plot correlation with real covariance matrix (top vs bottom)
[~, sort_idx]     	= sort(RSS);
ets_sorted          = edge_ts_all(:,sort_idx);
simil_top           = NaN(T,1);
simil_bot           = NaN(T,1);
mod_top             = NaN(T,1);
mod_bot             = NaN(T,1);
% same for null data
[~, sort_idx_est]   = sort(RSS_est_wishart_short);
ets_est_sorted      = ets_est_short(:,sort_idx_est);
simil_top_est       = NaN(T,1);
simil_bot_est       = NaN(T,1);
mod_top_est         = NaN(T,1);
mod_bot_est         = NaN(T,1);
for t = 1:T
    nCov_bot     	= NaN(N,N);
    nCov_bot(:)  	= mean(ets_sorted(:,1:t),2);
    nFC_bot         = corrcov(nCov_bot);
    r               = corrcoef(nFC,nFC_bot);
    simil_bot(t)    = r(1,2);
    [~,mod_bot(t)]  = community_louvain(nFC_bot,[],[],'negative_asym');
    nCov_top     	= NaN(N,N);
    nCov_top(:) 	= mean(ets_sorted(:,end-t+1:end),2);
    nFC_top         = corrcov(nCov_top);
    r               = corrcoef(nFC,nFC_top);
    simil_top(t)    = r(1,2);
    [~,mod_top(t)]  = community_louvain(nFC_top,[],[],'negative_asym');
    % same for null
    nCov_bot_est    = NaN(N,N);
    nCov_bot_est(:) = mean(ets_est_sorted(:,1:t),2);
    nFC_bot_est     = corrcov(nCov_bot_est);
    r               = corrcoef(nFC,nFC_bot_est);
    simil_bot_est(t)= r(1,2);
    [~,mod_bot_est(t)]  = community_louvain(nFC_bot_est,[],[],'negative_asym');
    nCov_top_est    = NaN(N,N);
    nCov_top_est(:) = mean(ets_est_sorted(:,end-t+1:end),2);
    nFC_top_est     = corrcov(nCov_top_est);
    r               = corrcoef(nFC,nFC_top_est);
    simil_top_est(t)= r(1,2);
    [~,mod_top_est(t)]  = community_louvain(nFC_top_est,[],[],'negative_asym');
    % correlation between FC estimates from real and null ets (sorted)
    r               = corrcoef(nFC_bot,nFC_bot_est);
    res.FC_sim_x_bot(t) = r(1,2);
    r               = corrcoef(nFC_top,nFC_top_est);
    res.FC_sim_x_top(t) = r(1,2);
end
% store FC similarity and modularity
res.FC_sim_top      = simil_top;
res.FC_sim_bot      = simil_bot;
res.FC_mod_top      = mod_top;
res.FC_mod_bot      = mod_bot;
res.FC_sim_top_est  = simil_top_est;
res.FC_sim_bot_est  = simil_bot_est;
res.FC_mod_top_est  = mod_top_est;
res.FC_mod_bot_est  = mod_bot_est;

%----------------------------------------------------------------------
% edge FC
eCov_diag           = NaN(N^2,1);
eCov_est_diag       = NaN(N^2,1);
[ix,iy]             = find(ones(N));
for ir = 1:N^2
    jj  = ix(ir);
    k   = iy(ir);
    eCov_est_diag(ir) = ...
        nCov(jj,jj) * nCov(k,k)...
        + nCov(jj,k) * nCov(k,jj) ...
        + nCov(jj,k) * nCov(jj,k);
    eCov_diag(ir)     = ...
        edge_ts_all(ir,:) * edge_ts_all(ir,:)' / (T - 1);
end
eCov_diag           = sqrt(eCov_diag);
eCov_est_diag       = sqrt(eCov_est_diag);
% quantities required for online Pearson correlation computation
% https://stats.stackexchange.com/questions/410468/online-update-of-pearson-coefficient
x_mean              = 0;
y_mean              = 0;
num                 = 0;
den1                = 0;
den2                = 0;
n                   = 1;
for ir = 1:N^2
    jj  = ix(ir);
    k   = iy(ir);
    for ic = ir+1:N^2 % only upper triangular
        l  = ix(ic);
        m  = iy(ic);
        eCov_est    = ...
            nCov(jj,l) * nCov(k,m)...
            + nCov(jj,m) * nCov(k,l) ...
            + nCov(jj,k) * nCov(l,m);
        eFC_est     = ...
            eCov_est / (eCov_est_diag(ir) * eCov_est_diag(ic));
        eCov        = ...
            edge_ts_all(ir,:) * edge_ts_all(ic,:)' / (T - 1);
        eFC         = eCov / (eCov_diag(ir) * eCov_diag(ic));
        % update quantities for online Pearson corr computation
        x_mean_old  = x_mean;
        y_mean_old  = y_mean;
        x_mean      = x_mean + (eFC - x_mean) / n;
        y_mean      = y_mean + (eFC_est - y_mean) / n;
        num         = num + (eFC - x_mean_old) * (eFC_est - y_mean);
        den1        = den1 + (eFC - x_mean_old) * (eFC - x_mean);
        den2        = den2 + (eFC_est - y_mean_old) * (eFC_est - y_mean);
        n           = n + 1;
    end
end
res.eFC_sim         = num / sqrt(den1 * den2);

% k-means
clusters_n = 10;
replications = 3;
[res.kmeans_idx,res.cent,res.sumdist] = kmeans(...
    eFC,clusters_n, ...
    'Replicates',replications,'Display','final');  
[res.kmeans_idx_est,res.cent_est,res.sumdist_est] = kmeans(...
    eFC_est,clusters_n, ...
    'Replicates',replications,'Display','final'); 

end
