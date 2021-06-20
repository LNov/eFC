%% Edge FC similarity
figure('Name','Edge FC Similarity', 'NumberTitle','off')
histogram(res.eFC_sim);
xlabel('Edge FC similarity (Pearson correlation)')

%% nFC similarity
figure('Name','nFC similarity', 'NumberTitle','off')
simil_top       = mean(res.FC_sim_top,'omitnan');
simil_bot       = mean(res.FC_sim_bot,'omitnan');
simil_top_est   = mean(res.FC_sim_top_est,'omitnan');
simil_bot_est   = mean(res.FC_sim_bot_est,'omitnan');
FC_sim_x_top    = mean(res.FC_sim_x_top,'omitnan');
FC_sim_x_bot    = mean(res.FC_sim_x_bot,'omitnan');

subplot(1,3,1)
labels = {'Top 5% RSS','Top 5% RSS (null)', ...
    'Bottom 5% RSS','Bottom 5% RSS (null)'};
[hdl,~] = fcn_boxpts([res.FC_sim_top(:,60),res.FC_sim_top_est(:,60),...
            res.FC_sim_bot(:,60),res.FC_sim_bot_est(:,60)],[],[]);
colors = ["#CB5032","#CB5032","#6077B3","#6077B3"];
for j = 1:4
   set(hdl.pointshandle{j},'markerfacecolor',colors(j),...
       'markerfacealpha',0.5); 
end
set(gca,'xtick',1:4,'xticklabel',labels);
xtickangle(40)
ylabel('Similarity to nFC')
pbaspect([1 1 1])
title('A')

subplot(1,3,2)
hold on
plot((1:T)/T,simil_top, ...
    'DisplayName', 'Descending RSS', ...
    'Color', '#CB5032')
plot((1:T)/T,simil_top_est, '--', ...
    'DisplayName', 'Descending RSS (null)', ...
    'Color', '#CB5032')
plot((1:T)/T,simil_bot, ...
    'DisplayName', 'Ascending RSS', ...
    'Color', '#6077B3')
plot((1:T)/T,simil_bot_est, '--', ...
    'DisplayName', 'Ascending RSS (null)', ...
    'Color', '#6077B3')
xlabel('Proportion of frames used')
ylabel('Similarity to nFC')
legend('Location','SouthEast')
pbaspect([1 1 1])
title('B')

subplot(1,3,3)
hold on
plot((1:T)/T,FC_sim_x_top, 'DisplayName', 'Descending RSS', ...
    'Color', '#CB5032')
xlabel('Proportion of frames used')
ylabel('Similarity to null model')
legend('Location','SouthEast')
pbaspect([1 1 1])
title('C')

%% modularity
figure('Name','FC modularity', 'NumberTitle','off')

subplot(1,2,1)
title('A')
labels = {'Top 5% RSS','Top 5% RSS (null)', ...
    'Bottom 5% RSS','Bottom 5% RSS (null)'};
[hdl,~] = fcn_boxpts([res.FC_mod_top(:,60),res.FC_mod_top_est(:,60),...
    res.FC_mod_bot(:,60),res.FC_mod_bot_est(:,60)],[],[]);
colors = ["#CB5032","#CB5032","#6077B3","#6077B3"];
for j = 1:4
   set(hdl.pointshandle{j},'markerfacecolor',colors(j),...
       'markerfacealpha',0.5); 
end
set(gca,'xtick',1:4,'xticklabel',labels);
xtickangle(45)
ylabel('Modularity')
pbaspect([1 1 1])


subplot(1,2,2)
title('B')
mod_top     = mean(res.FC_mod_top,'omitnan');
mod_bot     = mean(res.FC_mod_bot,'omitnan');
mod_top_est = mean(res.FC_mod_top_est,'omitnan');
mod_bot_est = mean(res.FC_mod_bot_est,'omitnan');

T_min       = 2;
mod_top     = mod_top(:,T_min:end);
mod_bot     = mod_bot(:,T_min:end);
mod_top_est = mod_top_est(:,T_min:end);
mod_bot_est = mod_bot_est(:,T_min:end);
hold on
plot((T_min:T)/T,mod_top, ...
    'DisplayName', 'Descending RSS', ...
    'Color', '#CB5032')
plot((T_min:T)/T,mod_top_est, '--', ...
    'DisplayName', 'Descending RSS (null)', ...
    'Color', '#CB5032')
plot((T_min:T)/T,mod_bot, ...
    'DisplayName', 'Ascending RSS', ...
    'Color', '#6077B3')
plot((T_min:T)/T,mod_bot_est, '--', ...
    'DisplayName', 'Ascending RSS (null)', ...
    'Color', '#6077B3')
xlabel('Proportion of frames used')
ylabel('Modularity')
legend('Location','southeast')
pbaspect([1 1 1])

%% PC1 coefficients

% Load HCP ICA-FIX time series for single subject (i_subj)
% using 'Schaefer2018_200Parcels_17Networks' parcellation
data = data - mean(data,1); % data is a (200 x 1200 x 100) array
data = data - mean(data,1); % GSR
data = zscore(data,0,2);    % z-score

% match (Faskowitz 2020) ROI order for Schaefer2018_200Parcels_17Networks
load('ROI_order.mat');
data(51:56,:,:)     = data(56:-1:51,:,:);   % move Limbic_B after Limbic_A
data(157:164,:,:)   = data(164:-1:157,:,:); % move Limbic_B after Limbic_A
data                = data(idx_sort,:,:);   % sort to match Faskowitz 2020

% concatenate time series for all subjects
data_subj   = squeeze(num2cell(data,[1 2])); % put each subject into a cell
data_cat    = cell2mat(data_subj');

% RSS and top percentiles
RSS         = permute(sum(data.^2,1),[2,3,1]);
perc        = 95;
top_idx     = RSS > prctile(RSS,perc,1);
bot_idx     = RSS < prctile(RSS,100-perc,1);

[U,S,V]     = svd(data_cat,'econ');
eigvals_sq  = diag(S).^2;
var_expl    = eigvals_sq(1) / sum(eigvals_sq); % variance explained by PC1
fprintf("Variance explained by PC1: %.0f%%\n",100*var_expl)
% select PC1 coeffs corresponding to top and bottom RSS values
PC1_top     = V(top_idx(:),1);
PC1_bot     = V(bot_idx(:),1);

% repeat for null
data_null   = NaN(size(data));
T           = size(data,2); % time steps
for i_subj = 1:subj_n
    % Parcellated time series
    ts = data(1:VOI_n,1:T,i_subj);
    % node covariance matrix
    ts_centered = ts - mean(ts,2);
    nCov        = (ts_centered * ts_centered') / (T - 1);
    % sample multivariate gaussian
    data_null(:,:,i_subj) = mvnrnd(zeros(1,VOI_n),nCov,T)';
end
% concatenate time series for all subjects
data_subj_null   = squeeze(num2cell(data_null,[1 2]));
data_cat_null    = cell2mat(data_subj_null');
% RSS and top percentiles
RSS_null         = permute(sum(data_null.^2,1),[2,3,1]);
top_idx_null     = RSS_null > prctile(RSS_null,perc,1);
top_idx_null     = top_idx_null(:);
bot_idx_null     = RSS_null < prctile(RSS_null,100-perc,1);
bot_idx_null     = bot_idx_null(:);
RSS_null         = RSS_null';
% PCA
[U_null,S_null,V_null]     = svd(data_cat_null,'econ');
eigvals_sq_null  = diag(S_null).^2;
var_expl_null    = eigvals_sq_null(1) / sum(eigvals_sq_null);
fprintf("Variance explained by PC1 (null): %.0f%%\n",100*var_expl_null)
% select PC1 coeffs corresponding to top and bottom RSS values
PC1_top_null     = V_null(top_idx_null,1);
PC1_bot_null     = V_null(bot_idx_null,1);


figure('Name','PC1 coefficients (absolute value)', 'NumberTitle','off')
labels = {'Top 5% RSS','Top 5% RSS (null)', ...
    'Bottom 5% RSS','Bottom 5% RSS (null)'};
[hdl,~] = fcn_boxpts([abs(PC1_top),abs(PC1_top_null), ...
    abs(PC1_bot),abs(PC1_bot_null)],[],[]);
colors = ["#CB5032","#CB5032","#6077B3","#6077B3"];
for j = 1:4
   set(hdl.pointshandle{j},'markerfacecolor',colors(j),...
       'markerfacealpha',0.5); 
end
set(gca,'xtick',1:4,'xticklabel',labels);
xtickangle(45)
ylabel("PC1 coefficients (absolute value)")

%% Leading FC eigenvector (z-score)
figure
addpath('Violinplot-Matlab')
% From Bechtold, Bastian, 2016. Violin Plots for Matlab, Github Project  
% https://github.com/bastibe/Violinplot-Matlab
% DOI: 10.5281/zenodo.4559847

pc1_zscore          = zscore(pc1);
pc1_zscore_null     = zscore(pc1_null);
pc1_zscore_th       = zscore(pc1_th);

pc1_scores          = NaN(VOI_n,1);
pc1_scores_null     = NaN(VOI_n,1);
pc1_scores_th       = NaN(VOI_n,1);
categories          = cell(VOI_n,1);
for i = 1:length(labels_short)
    idx                     = ticks_first(i):ticks_first(i+1)-1;
    categories(idx)         = labels_short(i);
    pc1_scores(idx)         = pc1_zscore(idx);
    pc1_scores_null(idx)    = pc1_zscore_null(idx);
    pc1_scores_th(idx)      = pc1_zscore_th(idx);
end

vs_th = violinplot(pc1_scores_th, categories,'GroupOrder', labels_short);
yline(0)
xlim([0,length(labels_short)+1])
xtickangle(90)
ylabel("Leading FC eigenvector (z-score)")

colors_custom = [...
    "#E8E800","#E8E800","#E8E800",...
    "#C00000","#C00000","#C00000",...
    "#6B9132","#6B9132",...
    "#D0D0D0",...
    "#66B2CC","#66B2CC","#66B2CC","#66B2CC",...
    "#CC99CC",...
    "#6270B2","#6270B2"];
for i = 1:length(labels_short)
    vs(i).ViolinColor = colors_custom(i);
    vs_null(i).ViolinColor = colors_custom(i);
    vs_th(i).ViolinColor = colors_custom(i);
end

%% Eigenvectors contributions to RSS
eigs_n          = 4;
FC_est_eigs     = NaN(VOI_n,VOI_n,eigs_n);
sim_FC_eigs     = NaN(subj_n,eigs_n,T);
sim_eigs        = NaN(subj_n,eigs_n,T);
sim_eigs_svd    = NaN(subj_n,eigs_n,T);

for i_subj = 1:subj_n
    RSS                 = res.RSS(i_subj,:);
    [~,idx_sort_RSS]    = sort(RSS,'descend');
    ts                  = data(:,idx_sort_RSS,i_subj);
    ts_centered         = ts - mean(ts,2);
    nCov_subj           = (ts_centered * ts_centered') / (T - 1);
    [eigvec,D]          = eig(nCov_subj);
    [~,ind]             = sort(diag(D),'descend'); % sort eigenvalues
    eigvec              = eigvec(:,ind);
    
    [~,~,V]             = svd(ts,'econ');
    sim_eigs_svd(i_subj,:,:) = abs(V(:,1:eigs_n)');
    
    for eig_i = 1:eigs_n
        FC_est_eigs(:,:,eig_i) = eigvec(:,eig_i) * eigvec(:,eig_i)';
        for t=1:T
            r = corrcoef(ts(:,t)*ts(:,t)',FC_est_eigs(:,:,eig_i));
            sim_FC_eigs(i_subj,eig_i,t) = r(1,2);
            r = corrcoef(ts(:,t),eigvec(:,eig_i));
            sim_eigs(i_subj,eig_i,t) = r(1,2);
        end
    end
end

figure
hold on
sim_eigs_mean = squeeze(mean(sim_FC_eigs,1));
sim_eigs_mean = movmean(sim_eigs_mean,40,2);
for eig_i = 1:eigs_n
    plot(sim_eigs_mean(eig_i,:),...
        'DisplayName', sprintf("%i",eig_i))
end
pbaspect([1 1 1])
xlabel("Frames (sorted by descending RSS)")
ylabel("Similarity to leading eigenvectors")
legend('Location','northeast')

figure
hold on
sim_eigs_mean = squeeze(mean(sim_eigs_svd,1));
sim_eigs_mean = movmean(sim_eigs_mean,40,2);
for eig_i = 1:eigs_n
    plot(sim_eigs_mean(eig_i,:),...
        'DisplayName', sprintf("PC %i",eig_i))
end
pbaspect([1 1 1])
xlabel("Frames (sorted by descending RSS)")
ylabel("PC coefficient (absolute value)")
legend('Location','northeast')
