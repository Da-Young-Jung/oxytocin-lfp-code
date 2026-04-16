% make 2D matrix (observation x features)
addpath Figure4
load('OT_LFP.mat')

data_sub = data(:,1:end,:, :,:);
tmp = permute(data_sub, [1 5 2 3 4]); % →> 15 × 84 × 6 x 3 x4
x = reshape(tmp, 15*84, size(data_sub,2)*3*4); % -> feature size = 1260 × 72
oxy = repelem(is_oxy(:), 15);
Nfeat = size(x, 2);

freq_labels = {'T1', 'T2', 'B1','B2', 'G1','G2'};
region_labels = {'mPFC', 'BLA', 'A1'};
value_labels = {'Power', 'PLV', 'GC1', 'GC2'};
feature_label = cell(1, numel(freq_labels)*numel(region_labels)*numel(value_labels));
idx = 1;

for v = 1:numel(value_labels)
    for r = 1:numel(region_labels)
        for f = 1:numel(freq_labels)
            feature_label{idx} = sprintf('%s-%s-%s', ...
                freq_labels{f}, region_labels{r}, value_labels{v});
            idx = idx + 1;
        end
    end
end
% normalize
mu = mean(x, 1, 'omitnan');
sig = std(x, 0, 1, 'omitnan');
valid = sig > 0;
Xz = (x(:, valid) - mu(valid)) ./ sig(valid);
%% find important featuers: stat test
for i = 1:Nfeat
    p(i) = kruskalwallis(Xz(:,i), oxy,'off');
end
[~,idx] = sort(-log10(p),'descend');

p_index = find(p<0.05);
length(find(p<0.05))/72*100;
feature_label(p_index);

N = 15;
valid_row = all(~isnan(Xz), 2);
x_clean = Xz(valid_row, :);
y_clean = oxy(valid_row);
lda = fitcdiscr(x_clean, y_clean, 'DiscrimType','linear');
cvlda = crossval(lda, 'KFold', 10);
accuracy = (1 - kfoldLoss(cvlda))*100
w = lda.Coeffs(1,2).Linear;   % class order 확인 후 사용
score = x_clean * w;
mean(score(y_clean==1))
mean(score(y_clean==0))
y_pred = score < 0; % 1 = oxy, 0 = saline
TP = sum((y_clean == 1) & (y_pred == 1));
FN = sum((y_clean == 1) & (y_pred == 0));
TN = sum((y_clean == 0) & (y_pred == 0));
FP = sum((y_clean == 0) & (y_pred == 1));

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy    = (TP + TN) / numel(y_clean)

fig = figure(1); clf;
set(fig, 'Color','w', 'Position',[291 365 700 417]);
subplot(2,3,1+3)

hold on
histogram(score(y_clean==0), 30, ...
    'Normalization','probability', ...
    'FaceAlpha',0.6, 'FaceColor',[1 1 1]*0.5);
histogram(score(y_clean==1), 30, ...
    'Normalization','probability', ...
    'FaceAlpha',0.6, 'FaceColor',[0.85 0.2 0.2]);
xlabel('LDA projection');
ylabel('Probability');
box off; hold off;
axis square
set(gca, ...
    'YTick', 0:0.04:0.12, ...
    'LineWidth',1, ...
    'FontSize',10, ...
    'Box','off');


%% AUC

Nboot = 50;        % bootstrap 반복 수
testFrac = 0.3;    % test 비율

N = size(x_clean,1);
d = size(x_clean,2);

Beta_all  = zeros(d, Nboot);
Beta0_all = zeros(1, Nboot);
AUC_all   = zeros(1, Nboot);

rng(1);

for b = 1:Nboot
    idx = randsample(N, N, true);
    cv = cvpartition(y_clean(idx), 'HoldOut', testFrac);

    Xtr = x_clean(idx(training(cv)), :);
    Xtr = (Xtr -mean(Xtr))./std(Xtr );
    ytr = y_clean(idx(training(cv)));

    Xte = x_clean(idx(test(cv)), :);
    yte = y_clean(idx(test(cv)));

    [B, FitInfo] = lassoglm(Xtr, ytr, 'binomial', 'CV', 5);

    k = FitInfo.IndexMinDeviance;
    beta  = B(:,k);
    beta0 = FitInfo.Intercept(k);

    score = beta0 + Xte * beta;
    p = 1 ./ (1 + exp(-score));
    [~,~,~,AUC] = perfcurve(yte, p, 1);

    % 저장
    Beta_all(1:numel(beta), b) = beta;
    Beta0_all(b) = beta0;
    AUC_all(b) = AUC;
end

% 좋은 거 고르자.
thr = prctile(AUC_all, 75);
goodModels = AUC_all >= thr;
beta_final  = median(Beta_all(:,goodModels), 2);
beta0_final = median(Beta0_all(goodModels));
logistic_fun = @(X) 1 ./ (1 + exp(-(beta0_final + X * beta_final)));

% 결과?
p_final = logistic_fun(x_clean);
[fp,tp,~,AUC_final] = perfcurve(y_clean, p_final, 1);

%%
%% AUC plot
subplot(2,3,2+3)
hold on
% plot(1-fp, tp, 'k-', 'LineWidth', 2);

plot(1-fp, tp, 'Color',[0.85 0.2 0.2], 'LineWidth',2);
axis square
xlim([0 1]); ylim([0 1]);

xlabel('Specificity');
ylabel('Sensitivity');

% ticks
set(gca, ...
    'XTick', 0:0.5:1, ...
    'YTick', 0:0.5:1, ...
    'LineWidth',1, ...
    'FontSize',10, ...
    'Box','off');
set(gca, 'XDir', 'reverse')
text(0.7, 0.15, sprintf('AUC = %.3f', AUC_final), ...
    'FontSize',10, 'FontWeight','normal');
hold off

%% PCA (unsupervised) for manifold view
[coeff, score, latent, ~, explained, mu_pca] = pca(x_clean);
% score: N x d, PC scores

isOxy = (y_clean == 1);
isSal = (y_clean == 0);

% (A) PCA scatter (PC1 vs PC2)
subplot(2,3,3+3); hold on

% points
scatter(score(isSal,1), score(isSal,2), 16, ...
    'filled', 'MarkerFaceColor',[0.6 0.6 0.6], 'MarkerFaceAlpha',0.6);
scatter(score(isOxy,1), score(isOxy,2), 16, ...
    'filled', 'MarkerFaceColor',[0.85 0.2 0.2], 'MarkerFaceAlpha',0.6);

% centroids (optional but nice)
cSal = mean(score(isSal,1:2), 1);
cOxy = mean(score(isOxy,1:2), 1);
plot(cSal(1), cSal(2), 'ko', 'MarkerSize',6, 'LineWidth',2);
plot(cOxy(1), cOxy(2), 'ko', 'MarkerSize',6, 'LineWidth',2);

% arrow between centroids (optional)
quiver(cSal(1), cSal(2), cOxy(1)-cSal(1), cOxy(2)-cSal(2), 0, ...
    'Color',[0 0 0], 'LineWidth',1, 'MaxHeadSize',0.8);

xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));

axis square
box off
set(gca, 'LineWidth',1, 'FontSize',10);
legend({'saline','oxytocin','centroids',''}, 'Location','best');
legend boxoff
