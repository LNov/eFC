function res = efc_single_subject(ts)
% Compute edge-centric features and compare with null model in terms of nFC
% similarity from top vs bottom RSS frames, as well as similarity between
% empirical and predicted eFC matrices.
% input: 2D time series array (n_parcels x n_frames)

N          	= size(ts,1); % number of parcels (regions)
T          	= size(ts,2); % number of frames
T_null      = T * 50;     % number of frames for null model

% Match (Faskowitz 2020) ROI order for Schaefer2018_200Parcels_17Networks
if N == 200
    load('ROI_order.mat','idx_sort');
    ts(51:56,:)     = ts(56:-1:51,:);   % move Limbic_B after Limbic_A
    ts(157:164,:)   = ts(164:-1:157,:); % move Limbic_B after Limbic_A
    ts              = ts(idx_sort,:);   % sort to match Faskowitz 2020
end

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
% eFC (empirical)
eFC                 = eye((N*(N-1))/2);
eCov_diag           = NaN((N*(N-1))/2,1);
for ir = 1:(N*(N-1))/2
    eCov_diag(ir)   = ...
        edge_ts(ir,:) * edge_ts(ir,:)' / (T - 1);
end
% take sqrt once here instead of repeatedly later
eCov_diag           = sqrt(eCov_diag);
for ir = 1:(N*(N-1))/2
    for ic = ir+1:(N*(N-1))/2 % only upper triangular
        eCov        = ...
            edge_ts(ir,:) * edge_ts(ic,:)' / (T - 1);
        eFC(ir,ic)  = eCov / (eCov_diag(ir) * eCov_diag(ic));
        eFC(ic,ir)  = eFC(ir,ic); % symmetric
    end
end
% eFC (null)
[ix,iy]             = find(triu(ones(N),1));
eFC_est             = eye(length(ix));
eCov_est_diag       = NaN(length(ix),1);
for ir = 1:length(ix)
    jj  = ix(ir);
    k   = iy(ir);
    eCov_est_diag(ir) = ...
        nCov(jj,jj) * nCov(k,k)...
        + nCov(jj,k) * nCov(k,jj) ...
        + nCov(jj,k) * nCov(jj,k);
end
% take sqrt once here instead of repeatedly later
eCov_est_diag       = sqrt(eCov_est_diag);
for ir = 1:length(ix)
    jj  = ix(ir);
    k   = iy(ir);
    for ic = ir+1:length(ix)
        l  = ix(ic);
        m  = iy(ic);
        eCov_est    = ...
            nCov(jj,l) * nCov(k,m)...
            + nCov(jj,m) * nCov(k,l) ...
            + nCov(jj,k) * nCov(l,m);
        eFC_est(ir,ic)     = ...
            eCov_est / (eCov_est_diag(ir) * eCov_est_diag(ic));
        eFC_est(ic,ir)  = eFC_est(ir,ic); % symmetric
    end
end
% correlation between empirical and null eFC
upper_tri           = logical(triu(ones((N*(N-1))/2),1));
r                   = corrcoef(eFC(upper_tri),eFC_est(upper_tri));
res.eFC_sim         = r(1,2);

end
