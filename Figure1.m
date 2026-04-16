clear; clc; close all;
addpath Figure1/
%% Load data
load('ERP.mat');
load('meta.mat', 'meta');

conds_to_analyze   = [1 2 3 4];    % Pup calls conditions
exclude_sessions   = [25, 39, 55, 56, 57, 58,59, 60, 67,68, 69,70]; % outlier sessions
n_mean     = 10;
offset_val = 30;
use_outlier = true;
th_z        = 3;
plot_time_range = [-0.1 0.4]; % sec
srate_onset = 300000;
n_sessions = size(ERP_all, 4);
drug = meta.drug_type(1:n_sessions);

%% Separate data by drugtype
n_sessions = size(ERP_all, 4);
valid_sessions = true(n_sessions,1);
valid_sessions(exclude_sessions) = false;

drug = meta.drug_type(1:n_sessions);
mask_ot = (drug == 1) & ~isnan(drug) & valid_sessions;
mask_sa = (drug ~= 1) & ~isnan(drug) & valid_sessions;

ids_oxt = find(mask_ot);
ids_sal = find(mask_sa);
n_oxt = length(ids_oxt);
n_sal = length(ids_sal);
target_N = min(n_oxt, n_sal);
ids_oxt_use = ids_oxt(1:target_N);
ids_sal_use = ids_sal(1:target_N);

% group info
groups(1).name = 'oxytocin';
groups(1).subject_eegIDs = ids_oxt_use;
groups(1).mask = mask_ot;
groups(2).name = 'saline';
groups(2).subject_eegIDs = ids_sal_use;
groups(2).mask = mask_sa;

pup_indices = find(ismember(trial_list_universal(:)', conds_to_analyze));

% Process data
for g = 1:numel(groups)
    current_ids = groups(g).subject_eegIDs;

    X = ERP_all(:,:,:, current_ids);
    m_subj_avg = mean(X, 4, 'omitnan');
    m_pup_data = m_subj_avg(:,:,pup_indices);

    [chN, T, N_pup_calls] = size(m_pup_data);
    m_pup_data_filt = m_pup_data;

    % Outlier Removal
    if use_outlier
        amp = squeeze(rms(m_pup_data, 2, 'omitnan'));
        mu_a = mean(amp, 2, 'omitnan');
        sd_a = std(amp, 0, 2, 'omitnan');
        z_amp = (amp - mu_a) ./ sd_a;
        out_mask = abs(z_amp) > th_z;
        for ch_idx = 1:chN
            bad_trials = find(out_mask(ch_idx,:));
            if ~isempty(bad_trials)
                m_pup_data_filt(ch_idx,:,bad_trials) = NaN;
            end
        end
    end

    % Block Averaging
    Ng = floor(N_pup_calls / n_mean);
    m_group = nan(chN, T, Ng);
    for blk = 1:Ng
        tr_idx = (blk-1)*n_mean + (1:n_mean);
        m_group(:,:,blk) = mean(m_pup_data_filt(:,:,tr_idx), 3, 'omitnan');
    end
    groups(g).data = m_group;
end

%% Visualization: stacked ERP (Figure 1C)
fprintf('Plotting results for channel: %s\n', chan_names{3});

fig = figure(1); clf;
set(fig, "Position" ,  [620 272 827 695], "Color", 'w')
chan_names = {'mPFC', 'BLA', 'AC', 'USV'};
load('StimulusInfo.mat', 'trialInfo');
for ch = 3
    trial_onset_universal = single(trialInfo.trial_onset(:));
    plot_ch_name = chan_names{ch};
    Ng_oxt = size(groups(1).data, 3);
    Ng_sal = size(groups(2).data, 3);
    Ng = min(Ng_oxt, Ng_sal);

    % Colormap
    if exist('customcolormap_preset','file')
        cmap0 = customcolormap_preset('red-yellow-blue');
        cmap  = flip(interp1(linspace(0,1,size(cmap0,1)), cmap0, linspace(0,1,Ng))*0.85);
    else
        cmap = parula(Ng);
    end

    % YTick
    onset_sec_all_pup_calls = trial_onset_universal(pup_indices) / srate_onset;
    onset_min_all_blocks = nan(1, Ng);

    for g_calc = 1:Ng
        tr_idx = (g_calc-1)*n_mean + 1;
        if tr_idx > length(onset_sec_all_pup_calls)
            Ng = g_calc - 1;
            break;
        end
        onset_min_all_blocks(g_calc) = floor(onset_sec_all_pup_calls(tr_idx) / 60);
    end
    onset_min_all_blocks = onset_min_all_blocks(1:Ng);

    target_labels = {'0', '10', '20', '30'};
    target_mins = [0, 10, 20, 30];
    yticks_final = [];
    yticklabels_final = {};

    for i = 1:length(target_mins)
        [~, g_idx] = min(abs(onset_min_all_blocks - target_mins(i)));
        current_tick_pos = -(g_idx - 1) * offset_val;
        if ~ismember(current_tick_pos, yticks_final)
            yticks_final(end+1) = current_tick_pos;
            yticklabels_final{end+1} = target_labels{i};
        end
    end

    xticks(0:0.2:0.4)
    [yticks, sort_idx] = sort(yticks_final);
    yticklabels = yticklabels_final(sort_idx);

    % --- Subplot 1: Oxytocin ---
    ax1 = subplot(5, 4, [1 5 9]);
    ax1.Position =[    0.1300    0.5461    0.1466    0.4212];

    hold(ax1, 'on');
    for blk = 1:Ng
        offset = (blk-1) * offset_val;
        plot(ax1, times, groups(1).data(ch,:,blk) - offset, ...
            'Color', cmap(min(blk, Ng),:), 'LineWidth', 1);
    end
    xline(ax1, 0, '--k', "LineWidth",1);
    xlabel(ax1, 'Time (s)');
    ylabel(ax1, 'Time from injection (min)');
    title(ax1, sprintf('%s | %s', plot_ch_name, groups(1).name));
    xlim(ax1, plot_time_range);
    axis(ax1, 'tight');
    box(ax1, 'off');
    set(ax1, "XTick", xticks, 'YTick', yticks, 'YTickLabel', yticklabels, 'YDir', 'normal');
    set(gca, "LineWidth",1)
    ylim([-800 200]);
    set(gca, 'XColor', 'none', 'Ycolor', 'none', 'FontSize',10)

    % --- Subplot 2: Saline ---
    ax2 = subplot(5, 4, [2 6 10]);
    ax2.Position =  [0.3075    0.5461    0.1466    0.4212];
    hold(ax2, 'on');
    for blk = 1:Ng
        offset = (blk-1) * offset_val;
        plot(ax2, times, groups(2).data(ch,:,blk) - offset, ...
            'Color', cmap(min(blk, Ng),:), 'LineWidth', 1.2);
    end
    xline(ax2, 0, '--k', "LineWidth",1);
    xlabel(ax2, 'Time (s)');
    title(ax2, sprintf('%s | %s', plot_ch_name, groups(2).name));
    xlim(ax2, plot_time_range);
    axis(ax2, 'tight');
    box(ax2, 'off');
    set(ax2, "XTick", xticks, 'YTick', yticks, 'YTickLabel', yticklabels, 'YDir', 'normal');
    set(gca, "LineWidth",1, 'FontSize',10)

    % Link Axes
    linkaxes([ax1, ax2]);
    xlim([ax1, ax2], plot_time_range);
    set(gca, 'XColor', 'none', 'Ycolor', 'none')
    ylim([-670 50]);
end

% Colorbar
cbar = colorbar('Location', 'southoutside');
cbar.Position = [0.1777    0.5737    0.1018    0.0119];
cbar.FontSize = 10;
cbar.LineWidth = 1;
ylabel(cbar, 'Time from injection (min)');
colormap(cmap);
caxis([1 Ng]);
cbar_ticks = [];
cbar_labels = {};
min_time_shown = onset_min_all_blocks(1);
max_time_shown = onset_min_all_blocks(Ng);

for i = 1:length(target_mins)
    target_min = target_mins(i);
    [~, g_idx] = min(abs(onset_min_all_blocks - target_min));
    cbar_ticks(end+1) = g_idx;
    cbar_labels{end+1} = target_labels{i};
end

if ~ismember(1, cbar_ticks)
    cbar_ticks = [1, cbar_ticks];
    cbar_labels = {num2str(floor(min_time_shown)), cbar_labels{:}};
end
if ~ismember(Ng, cbar_ticks)
    cbar_ticks = [cbar_ticks, Ng];
    cbar_labels = {cbar_labels{:}, num2str(floor(max_time_shown))};
end

[cbar_ticks, unique_idx] = unique(cbar_ticks);
cbar_labels = cbar_labels(unique_idx);
set(cbar, 'Ticks', cbar_ticks, 'TickLabels', cbar_labels);

plot([0.16 0.26], [1 1]*-650, 'k-', 'LineWidth', 2);
plot([0.26 0.26], [-650 -600], 'k-', 'LineWidth', 2);
text(0.23, -675, '100 ms', 'HorizontalAlignment', 'center', 'FontSize', 10)
text(0.338, -627, '50 \muV', 'HorizontalAlignment', 'center', 'FontSize', 10)

%% Figure 1D
time_start_minutes = 5;
time_end_minutes   = 20;
time_start_sec = time_start_minutes * 60;
time_end_sec   = time_end_minutes * 60;


trial_onset_sec = trial_onset_universal / srate_onset;
if ~exist('pup_indices', 'var')
    pup_call_mask = ismember(trial_list_universal(:), conds_to_analyze);
else
    pup_call_mask = false(size(trial_list_universal));
    pup_call_mask(pup_indices) = true;
end
time_mask = (trial_onset_sec(:) >= time_start_sec) & (trial_onset_sec(:) <= time_end_sec);
final_trial_indices = find(pup_call_mask & time_mask);

fprintf('  Found %d pup call trials within %d to %d minutes.\n', ...
    numel(final_trial_indices), time_start_minutes, time_end_minutes);


subject_averages_all_ch = cell(4, numel(groups));
for ch = 1:4
    for g = 1:numel(groups)
        current_mask = groups(g).mask;
        n_subj = sum(current_mask);
        X = ERP_all(:,:,:,current_mask);
        X_filtered = X(:,:,final_trial_indices,:);

        data_to_avg = squeeze(X_filtered(ch,:,:,:));
        if n_subj == 1 && ismatrix(data_to_avg)
            subj_means = mean(data_to_avg, 2, 'omitnan');
        else
            subj_means = mean(data_to_avg, 2, 'omitnan');
        end
        subject_averages_all_ch{ch, g} = squeeze(subj_means);
    end
end

sig_y_level_pos = 30;
sig_y_level_neg = -35;
stat_alpha = 0.05;

for ch = [1 3 2] % A1 -> mPFC -> BLA
    switch ch
        case 1 % mPFC
            pos_vec = [0.55  0.8288  0.1061  0.13];

        case 3 % A1
            pos_vec = [0.70  0.8288  0.1061  0.13];

        case 2 % BLA
            pos_vec = [0.85  0.8288  0.1061  0.13];
    end

    ax = subplot('Position', pos_vec);
    N_T = length(times);
    grand_avg_data = nan(2, N_T);
    grand_sem_data = nan(2, N_T);

    data_oxt_raw = subject_averages_all_ch{ch, 1};
    data_sal_raw = subject_averages_all_ch{ch, 2};

    bad_oxt_mask = any(isnan(data_oxt_raw), 1);
    bad_sal_mask = any(isnan(data_sal_raw), 1);
    data_oxt_good = data_oxt_raw(:, ~bad_oxt_mask);
    data_sal_good = data_sal_raw(:, ~bad_sal_mask);
    N_oxt_good = size(data_oxt_good, 2);
    N_sal_good = size(data_sal_good, 2);

    if N_sal_good < N_oxt_good
        N_to_use = N_sal_good;
        oxt_idx_keep = 1:N_to_use;
        sal_idx_keep = 1:N_sal_good;
    else
        N_to_use = N_oxt_good;
        oxt_idx_keep = 1:N_oxt_good;
        sal_idx_keep = 1:N_to_use;
    end

    fprintf('  Channel %s: N=%d (Matched).\n', chan_names{ch}, N_to_use);

    data_oxt_final = data_oxt_good(:, oxt_idx_keep);
    data_sal_final = data_sal_good(:, sal_idx_keep);

    % 4. Mean/SEM
    grand_avg_data(1, :) = mean(data_oxt_final, 2);
    grand_avg_data(2, :) = mean(data_sal_final, 2);
    grand_sem_data(1, :) = std(data_oxt_final, 0, 2) / sqrt(N_to_use);
    grand_sem_data(2, :) = std(data_sal_final, 0, 2) / sqrt(N_to_use);

    % 5. stat
    h_stats = nan(1, N_T);
    for t = 1:N_T
        [~, h_stats(t)] = ranksum(data_oxt_final(t, :), data_sal_final(t, :), 'alpha', stat_alpha);
    end

    sig_bar_red = nan(1, N_T);
    sig_bar_black = nan(1, N_T);

    % Mean
    mean_oxt = grand_avg_data(1, :);
    mean_sal = grand_avg_data(2, :);

    sig_mask_red   = (h_stats == 1) & (mean_oxt > mean_sal);
    sig_mask_black = (h_stats == 1) & (mean_sal > mean_oxt);

    sig_bar_red(sig_mask_red) = sig_y_level_pos;
    sig_bar_red((times<0)) = nan;

    if ch==3; sig_bar_black(sig_mask_black) = -78; else
        sig_bar_black(sig_mask_black) = sig_y_level_neg;end

    % Plot
    hold on;
    color_oxt = [0.85 0.2 .2];
    color_sal = [0.3 0.3 0.3];

    lineProps_oxt = {'Color', color_oxt, 'LineWidth', 1};
    lineProps_sal = {'Color', color_sal, 'LineWidth', 1};
    transparent_flag = 0.5;

    if exist('shadedErrorBar', 'file')
        h(1) = shadedErrorBar(times, grand_avg_data(1, :), grand_sem_data(1, :), lineProps_oxt, transparent_flag);
        h(2) = shadedErrorBar(times, grand_avg_data(2, :), grand_sem_data(2, :), lineProps_sal, transparent_flag);
        set(h(1).edge, 'LineStyle', 'none');
        set(h(2).edge, 'LineStyle', 'none');
        legend_handles = [h(1).mainLine, h(2).mainLine];
    else
        h1 = plot(times, grand_avg_data(1, :), 'Color', color_oxt);
        h2 = plot(times, grand_avg_data(2, :), 'Color', color_sal);
        legend_handles = [h1, h2];
    end

    plot(times, sig_bar_red, 'Color', color_oxt, 'LineWidth', 2);
    plot(times, sig_bar_black, 'Color', color_oxt, 'LineWidth', 2);

    xline(0, '--k');
    xlim(plot_time_range);
    xlabel('Time (s)');

    if ch==1; ylabel('Amplitude (\muV)');end
    title(sprintf('%s', chan_names{ch}));
    if ch==1
        lgd = legend(legend_handles, {groups(1).name, groups(2).name}, 'Location', 'northeast', 'Box','off');
        lgd.ItemTokenSize = [5, 12];
        lgd.Position = [    0.6155    0.9171    0.0611    0.0381];
    end

    box off;
    axis tight;
    xlim([-0.05 0.2]);
    if ch==3; ylim([-85 40]);
    else ylim([-40 40]); end
    set(gca, "LineWidth", 1, "YTick", -80:40:40, 'FontSize', 10);
    axis square
end 

%% Figure 1E
if ~exist('subject_averages_all_ch', 'var') || ~exist('times', 'var')
    error('Data (subject_averages_all_ch or times) not found. Run Section 5 (analysis part) first.');
end

baseline_time_window = [-2 0];
baseline_idx = (times >= baseline_time_window(1)) & (times <= baseline_time_window(2));
conversion_factor = 1e6; % uV^2 -> mV^2
color_oxt = [0.85 0.2 .2]; % Oxytocin: Red
color_sal = [0.3 0.3 0.3]; % Saline: Grey
marker_size_mean = 5;
line_width = 1;
font_size_axis = 10;
font_size_title = 10;
stat_alpha = 0.05;

% Subplot
for ch = [1 3 2] % A1 -> mPFC -> BLA 순서
    switch ch
        case 3
            ch_name = 'AC';
            pos_vec = [0.70  0.55   0.1061  0.13];
        case 1
            ch_name = 'mPFC';
            pos_vec = [0.55  0.55  0.1061  0.13];
        case 2
            ch_name = 'BLA';
            pos_vec = [0.85  0.55   0.1061  0.13];
    end
    ax = subplot('Position', pos_vec);
    hold(ax, 'on');

    if ch == 3
        response_time_window = [0.014 0.024];
    else
        response_time_window = [0 0.2];
    end
    response_idx_ch = (times >= response_time_window(1)) & (times <= response_time_window(2));
    data_oxt_raw = subject_averages_all_ch{ch, 1};
    data_sal_raw = subject_averages_all_ch{ch, 2};

    bad_oxt_mask = any(isnan(data_oxt_raw), 1);
    bad_sal_mask = any(isnan(data_sal_raw), 1);

    data_oxt_good = data_oxt_raw(:, ~bad_oxt_mask);
    data_sal_good = data_sal_raw(:, ~bad_sal_mask);

    N_final = min(size(data_oxt_good, 2), size(data_sal_good, 2));

    % Calc power
    if N_final > 0
        power_sig_oxt = mean(data_oxt_good(response_idx_ch, 1:N_final).^2, 1) / conversion_factor;
        power_sig_sal = mean(data_sal_good(response_idx_ch, 1:N_final).^2, 1) / conversion_factor;
        power_noise_oxt = mean(data_oxt_good(baseline_idx, 1:N_final).^2, 1) / conversion_factor;
        power_noise_sal = mean(data_sal_good(baseline_idx, 1:N_final).^2, 1) / conversion_factor;

        % Wilcoxon rank-sum test
        p_sig = ranksum(power_sig_oxt, power_sig_sal);
        p_noise = ranksum(power_noise_oxt, power_noise_sal);
        m_sig_oxt = median(power_sig_oxt, 'omitnan'); m_noise_oxt = median(power_noise_oxt, 'omitnan');
        m_sig_sal = median(power_sig_sal, 'omitnan'); m_noise_sal = median(power_noise_sal, 'omitnan');

        s_sig_oxt = 0.5 * std(power_sig_oxt)/sqrt(N_final); s_noise_oxt = 0.5 * std(power_noise_oxt)/sqrt(N_final);
        s_sig_sal = 0.5 * std(power_sig_sal)/sqrt(N_final); s_noise_sal = 0.5 * std(power_noise_sal)/sqrt(N_final);
    end

    h_oxt = plot(ax, [2.1, 1.1], [m_sig_oxt, m_noise_oxt], '-o', 'Color', color_oxt, ...
        'MarkerFaceColor', 'none', 'MarkerSize', marker_size_mean, 'LineWidth', line_width);
    errorbar(ax, [2.1, 1.1], [m_sig_oxt, m_noise_oxt], [s_sig_oxt, s_noise_oxt], ...
        'LineStyle', 'none', 'Color', color_oxt, 'LineWidth', line_width, 'CapSize', 8);

    h_sal = plot(ax, [1.9, 0.9], [m_sig_sal, m_noise_sal], '-o', 'Color', color_sal, ...
        'MarkerFaceColor', 'none', 'MarkerSize', marker_size_mean, 'LineWidth', line_width);
    errorbar(ax, [1.9, 0.9], [m_sig_sal, m_noise_sal], [s_sig_sal, s_noise_sal], ...
        'LineStyle', 'none', 'Color', color_sal, 'LineWidth', line_width, 'CapSize', 8);

    set(ax, 'YScale', 'log', 'LineWidth',1);

    if ch == 1
        lgd = legend([h_oxt, h_sal], {'oxytocin', 'saline'}, 'Location', 'southeast', 'Box', 'off', 'FontSize', 9);
        lgd.ItemTokenSize = [5, 12]; 
        lgd.Position = [0.6155 0.5524 0.0611 0.0381];
    end
    ylim(ax, [10^-5 0.01]);
    set(ax, 'YTick', [10^-6, 10^-5, 10^-4, 10^-3, 10^-2], 'YMinorTick', 'on', 'Box', 'off');
    grid(ax, 'on');

    star_y_pos = 0.002; 
    if p_sig < stat_alpha
        text(ax, 2, star_y_pos, '*', 'HorizontalAlignment', 'center', 'FontSize', 16);
    end
    if p_noise < stat_alpha
        text(ax, 1, star_y_pos, '*', 'HorizontalAlignment', 'center', 'FontSize', 16);
    end

    xlim(ax, [0.5 2.5]);
    set(ax, 'XTick', [1 2], 'XTickLabel', {'Pre-stim', 'Stim'}, 'FontSize', font_size_axis);

    if ch == 1
        ylabel(ax, 'Total power (mV^2)', 'FontSize', font_size_axis);
    else
        set(ax, 'YTickLabel', []);
    end

    title(ax, ch_name, 'FontSize', font_size_title);
    axis(ax, 'square');
    hold(ax, 'off');
end
fprintf('Individual scatter + Mean line plot (Section 19) complete.\n');


%% ASSR PLOTS
load('ITPC.mat');  % ITPC_full
ERP_times_full = ERP.times;

time_range_itpc = [-1 2];
idx_time_itpc = find(ERP_times_full >= time_range_itpc(1) & ERP_times_full <= time_range_itpc(2));
ERP_times_itpc = ERP_times_full(idx_time_itpc);
ERP_times_itpc = ERP_times_itpc(1:size(ITPC_full, 2));

drug_type = meta.drug_type(1:105);
include_control = find(drug_type == 0 | drug_type == -1);
include_drug = find(drug_type == 1);

n_control = length(include_control);
n_drug = length(include_drug);

target_n = 34;
include_control = include_control(1:target_n);
include_drug = include_drug(1:target_n);

%% Figure 1F
target_channels = [3, 1, 2];
channel_names   = {'AC', 'mPFC', 'BLA'};
target_freq_idx = 5;
stim_to_use     = [5];
plot_time_win   = [-0.2 1.2];
smooth_win      = 50;
cols = {[0.85 0.2 .2], [0.3 0.3 0.3]};

base_win = [-0.5 0];
base_idx = ERP_times_itpc >= base_win(1) & ERP_times_itpc < base_win(2);
t_idx_plot = ERP_times_itpc >= plot_time_win(1) & ERP_times_itpc <= plot_time_win(2);
times_plot = ERP_times_itpc(t_idx_plot);
target_indices = find(t_idx_plot);

for i = 1:length(target_channels)
    ch_idx = target_channels(i);
    ch_name = channel_names{i};

    switch i
        case 2 
            pos_vec = [0.55  0.04  0.1061  0.13];
        case 1 
            pos_vec = [0.70  0.04  0.1061  0.13];
        case 3
            pos_vec = [0.85  0.04  0.1061  0.13];
    end

    ax = subplot('Position', pos_vec); hold on;


    data_step1 = squeeze(ITPC_full(ch_idx, :, :, target_freq_idx, :));
    if isempty(stim_to_use)
        data_step2 = squeeze(mean(data_step1, 2, 'omitnan'));
    else
        if length(stim_to_use) > 1
            data_step2 = squeeze(mean(data_step1(:, stim_to_use, :), 2, 'omitnan'));
        else
            data_step2 = squeeze(data_step1(:, stim_to_use, :));
        end
    end

    itc_raw_groups = cell(1, 2);
    itc_raw_groups{1} = data_step2(:, include_drug)';
    itc_raw_groups{2} = data_step2(:, include_control)';
    stats_data = cell(1, 2);

    for g = 1:2
        raw_d = itc_raw_groups{g};
        bs_vals = mean(raw_d(:, base_idx), 2, 'omitnan');
        norm_d = raw_d - bs_vals;
        plot_d = norm_d(:, t_idx_plot);
        if smooth_win > 1, plot_d = movmean(plot_d, smooth_win, 2); end
        stats_data{g} = plot_d;

        mu = mean(plot_d, 1, 'omitnan');
        sem = std(plot_d, 0, 1, 'omitnan') / sqrt(size(plot_d, 1));

        fill([times_plot fliplr(times_plot)], [mu+sem fliplr(mu-sem)], ...
            cols{g}, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
        plot(times_plot, mu, '-', 'Color', cols{g}, 'LineWidth', 1);
    end

    p_vals = nan(1, length(target_indices));
    for k = 1:length(target_indices)
        d1 = stats_data{1}(:, k); d2 = stats_data{2}(:, k);
        if ~isempty(d1) && ~isempty(d2), p_vals(k) = ranksum(d1, d2); end
    end

    bar_y_pos = 0;
    if i==1, ylim([-0.1 1]); bar_y_pos = 0.7;
    else ylim([-0.05 0.5]); bar_y_pos = 0.35; end

    % Significant Bar
    sig_idx = p_vals < 0.05;
    if any(sig_idx)
        bar_y = bar_y_pos;
        lbl = bwlabel(sig_idx);
        for k = 1:max(lbl)
            seg_times = times_plot(lbl == k);
            if length(seg_times) > 1
                plot([seg_times(1) seg_times(end)], [bar_y bar_y], ...
                    '-', 'Color', cols{1}, 'LineWidth', 2);
            end
        end
    end

    xline(0, '--k'); xlabel('Time (s)');
    if i==2, ylabel('40 Hz ITPC');     end
    set(gca, 'YTick', [0 0.5 1])

    title(ch_name); xlim(plot_time_win);
    set(gca, 'LineWidth', 1, 'Box', 'off', 'TickDir', 'out', 'FontSize',10);

    if i == 2
        lgd = legend({'', 'oxytocin', '', 'saline'}, 'Location', 'best', 'Box', 'off');
        lgd.ItemTokenSize = [5, 12];
        lgd.Position = [    0.6155    0.1365    0.0611    0.0381];
        lgd.FontSize = 9;
    end
    axis square
end
fprintf('Figure 1F Generated with custom positions.\n');


%% Figure 1G
load ASSR.mat;
ASSR_ERP.PSD.data(:,:,:,[3,5,32]) = nan;

drug_vec = ASSR_ERP.drug_type;
mask_oxt = (drug_vec == 1);
mask_sal = (drug_vec == 0 | drug_vec == -1);

xlims = [-.500 1.500];
ylims_spec = [30 70];
caxis_spec = [0 5];
resize_factor = 3;
chan_label = {'mPFC', 'BLA', 'AC'};
group_masks = {mask_oxt, mask_sal};
group_names = {'oxytocin', 'saline'};

for g = 1:2
    if g == 1
        pos_vec = [0.15  0.042  0.13  0.13];
    else
        pos_vec = [0.30  0.042  0.15  0.13];
    end

    ax = subplot('Position', pos_vec);
    hold on;

    current_mask = group_masks{g};
    chanIdx = 1; % mPFC


    group_data = ASSR_ERP.PSD.data(:,:,chanIdx, current_mask);
    d = nanmean( abs(group_data), 4);
    d = d.^2;
    d = imresize(d, resize_factor);
    imagesc(ASSR_ERP.PSD.t, ASSR_ERP.PSD.f, d');
    axis xy;
    colormap(ax, jet(256));
    caxis(ax, caxis_spec);

    % 3. ERP Trace
    group_erp = ASSR_ERP.data(chanIdx, :, current_mask);
    erp_trace = nanmean(group_erp, 3);
    plot(ASSR_ERP.times, 55 + erp_trace*.14, '-', 'LineWidth', 0.5, 'Color', [1 1 1]*.9);
    plot([0 0], ylims_spec, 'w--', 'LineWidth', 0.8);
    plot([1 1], ylims_spec, 'w--', 'LineWidth', 0.8);

    ylim(ylims_spec); xlim(xlims);
    xlabel('Time (s)');
    if g == 1, ylabel('Freq (Hz)'); end

    % Colorbar
    if g == 2
        cb = colorbar(ax);
        ylabel(cb, 'Power (mV^2)');
        cb.FontSize = 10;
        cb.LineWidth = 1;
    end
    clim([0 6])
    set(ax, 'FontSize', 10, 'Box', 'off', 'LineWidth', 1);
    title(ax, sprintf('%s | %s', chan_label{chanIdx}, group_names{g}));
    xticks(0:1); clear yticks;
    yticks([30 50 70]);
    axis square
end
fprintf('ASSR Spectrogram Generated with custom positions.\n');

%% Figure 1H
color_oxt = [0.85 0.2 .2];
color_sal = [0.3 0.3 0.3];
marker_size_mean = 5; 
line_width = 1;     
stat_alpha = 0.05;
freq_range_idx = hb_findIdx([22 43], ASSR_ERP.PSD.f);
time_pre_idx = hb_findIdx([-3 -2], ASSR_ERP.PSD.t);
time_stim_idx = hb_findIdx([0.1 0.5], ASSR_ERP.PSD.t);

chan_label = {'mPFC','BLA','AC'};
include_sal = find(ASSR_ERP.drug_type == 0 | ASSR_ERP.drug_type == -1);
include_oxy = find(ASSR_ERP.drug_type == 1);

plot_order = [3, 1, 2];

for i = 1:3
    target_ch = plot_order(i);

    switch i
        case 2 
            pos_vec = [0.55  0.3  0.1061  0.13];
        case 1 
            pos_vec = [0.70  0.30  0.1061  0.13];
        case 3 
            pos_vec = [0.85  0.3  0.1061  0.13];
    end

    ax = subplot('Position', pos_vec); hold on;

    d_sal_pre_raw = squeeze(mean(mean(abs(ASSR_ERP.PSD.data(time_pre_idx, freq_range_idx, target_ch, include_sal)).^2, 1, 'omitnan'), 2, 'omitnan'));
    d_oxy_pre_raw = squeeze(mean(mean(abs(ASSR_ERP.PSD.data(time_pre_idx, freq_range_idx, target_ch, include_oxy)).^2, 1, 'omitnan'), 2, 'omitnan'));
    d_sal_stim_raw = squeeze(mean(mean(abs(ASSR_ERP.PSD.data(time_stim_idx, freq_range_idx, target_ch, include_sal)).^2, 1, 'omitnan'), 2, 'omitnan'));
    d_oxy_stim_raw = squeeze(mean(mean(abs(ASSR_ERP.PSD.data(time_stim_idx, freq_range_idx, target_ch, include_oxy)).^2, 1, 'omitnan'), 2, 'omitnan'));

    d_sal_pre = d_sal_pre_raw(d_sal_pre_raw < 100 & ~isnan(d_sal_pre_raw));
    d_oxy_pre = d_oxy_pre_raw(d_oxy_pre_raw < 100 & ~isnan(d_oxy_pre_raw));
    d_sal_stim = d_sal_stim_raw(d_sal_stim_raw < 100 & ~isnan(d_sal_stim_raw));
    d_oxy_stim = d_oxy_stim_raw(d_oxy_stim_raw < 100 & ~isnan(d_oxy_stim_raw));

    m_sal_pre = mean(d_sal_pre); sem_sal_pre = std(d_sal_pre)/sqrt(length(d_sal_pre));
    m_oxy_pre = mean(d_oxy_pre); sem_oxy_pre = std(d_oxy_pre)/sqrt(length(d_oxy_pre));
    m_sal_stim = mean(d_sal_stim); sem_sal_stim = std(d_sal_stim)/sqrt(length(d_sal_stim));
    m_oxy_stim = mean(d_oxy_stim); sem_oxy_stim = std(d_oxy_stim)/sqrt(length(d_oxy_stim));

    x_sal = [0.9, 1.9]; x_oxy = [1.1, 2.1];

    % Saline:  ('MarkerFaceColor', 'none' )
    h_sal = plot(ax, x_sal, [m_sal_pre, m_sal_stim], '-o', 'Color', color_sal, ...
        'MarkerFaceColor', 'none', 'MarkerSize', marker_size_mean, 'LineWidth', line_width);
    errorbar(ax, x_sal, [m_sal_pre, m_sal_stim], [sem_sal_pre/2, sem_sal_stim/2], ...
        'Color', color_sal, 'LineStyle', 'none', 'LineWidth', line_width, 'CapSize', 8, 'HandleVisibility', 'off');

    % Oxytocin:  ('MarkerFaceColor', 'none' )
    h_oxt = plot(ax, x_oxy, [m_oxy_pre, m_oxy_stim], '-o', 'Color', color_oxt, ...
        'MarkerFaceColor', 'none', 'MarkerSize', marker_size_mean, 'LineWidth', line_width);
    errorbar(ax, x_oxy, [m_oxy_pre, m_oxy_stim], [sem_oxy_pre/2, sem_oxy_stim/2], ...
        'Color', color_oxt, 'LineStyle', 'none', 'LineWidth', line_width, 'CapSize', 8, 'HandleVisibility', 'off');

    % --- 3. Style Legend ---
    set(gca, 'YScale', 'log', 'FontSize', 10, 'Box', 'off', 'TickDir', 'in', 'LineWidth', 1);
    xlim([0.5 2.5]); ylim([3*10^-2 50]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'Pre-stim', 'Stim'}, ...
        'YTick', [10^-1 10^0 10^1], 'YMinorTick', 'on');
    grid on; axis square;

    if i == 2 % mPFC 
        ylabel('40 Hz Power (\muV^2)');
        lgd = legend([h_oxt, h_sal], {'oxytocin', 'saline'}, 'Location', 'none', 'Box', 'off');
        lgd.FontSize = 9;
        lgd.ItemTokenSize = [5, 12];
        lgd.Position = [0.6155, 0.3092, 0.0611, 0.0381];
    else
        set(gca, 'YTickLabel', []);
    end

    title(chan_label{target_ch}, 'FontSize', font_size_title, 'FontWeight', 'normal');
    p_pre = ranksum(d_oxy_pre, d_sal_pre);
    p_stim = ranksum(d_oxy_stim, d_sal_stim);
    y_star = 15;
    if p_pre < stat_alpha, text(1, y_star, '*', 'HorizontalAlignment', 'center', 'FontSize', 16); end
    if p_stim < stat_alpha, text(2, y_star, '*', 'HorizontalAlignment', 'center', 'FontSize', 16); end

    hold(ax, 'off');
end
