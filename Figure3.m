%% Figure 3
clear; clc;
addpath Figure3/
%% Load data
load('plv.mat'); 
load('psd_time.mat');
load('meta.mat');

fps = 30;
inj_time = [meta.inj_start meta.inj_stop] / fps;

%% Parameters
ch_names = {'mPFC-BLA','mPFC-AC','BLA-AC'};
n_ch = numel(ch_names);

band_def = struct( ...
    'thetalo', [4 8], ...
    'thetahi', [8 12], ...
    'betalo',  [18 24], ...
    'betahi',  [24 32], ...
    'gammalo', [30 50], ...
    'gammahi', [70 90]);

b_names = {'\theta_{low}','\theta_{high}', ...
           '\beta_{low}','\beta_{high}', ...
           '\gamma_{low}','\gamma_{high}'};

band_names = fieldnames(band_def);
n_band = numel(band_names);

time_range = [-1200 3600];
t_common = (time_range(1):1:time_range(2))';
mask_0_20 = t_common >= 0 & t_common < 20 * 60;

sal_idx = find(drug == 0);
oxy_idx = find(drug == 1);

deltaMat = nan(n_band, n_ch);
pMat = nan(n_band, n_ch);
sigMask = false(n_band, n_ch);

get_freq = @(P) ...
    (exist('freq','var') && ~isempty(freq) && numel(freq) >= size(P,2)) .* freq(1:size(P,2)) + ...
    (~(exist('freq','var') && ~isempty(freq) && numel(freq) >= size(P,2))) .* (1:size(P,2));

%% PLV: band x channel statistics
for b = 1:n_band
    fr = band_def.(band_names{b});

    for ch = 1:n_ch
        vals_sal = [];
        vals_oxy = [];

        for subj = 1:size(plv,1)
            if subj > numel(drug) || drug(subj) == -1
                continue;
            end
            if isempty(plv{subj,1}) || isempty(plv{subj,2})
                continue;
            end

            P = plv{subj,1};
            t = plv{subj,2};

            if size(P,3) < ch
                continue;
            end

            pow = P(:,:,ch);
            f = get_freq(P);
            f_idx = f >= fr(1) & f <= fr(2);

            if ~any(f_idx)
                continue;
            end

            band_series = mean(pow(:, f_idx), 2, 'omitnan');

            t_inj_start = inj_time(subj,1);
            t_inj_stop  = inj_time(subj,2);

            base_idx = t < t_inj_start;
            base = band_series(base_idx);

            if nnz(isfinite(base)) < 10 || std(base, 'omitnan') == 0
                continue;
            end

            z = (band_series - mean(base, 'omitnan')) / std(base, 'omitnan');

            t_align = t(:) - t_inj_stop;
            z_interp = interp1(t_align, z, t_common, 'linear', nan);

            if nnz(isfinite(z_interp(mask_0_20))) < 10
                continue;
            end

            z_0_20 = mean(z_interp(mask_0_20), 'omitnan');

            if drug(subj) == 0
                vals_sal(end+1,1) = z_0_20; %#ok<AGROW>
            elseif drug(subj) == 1
                vals_oxy(end+1,1) = z_0_20; %#ok<AGROW>
            end
        end

        if numel(vals_sal) >= 3 && numel(vals_oxy) >= 3
            p = ranksum(vals_sal, vals_oxy);
            pMat(b,ch) = p;
            deltaMat(b,ch) = mean(vals_oxy, 'omitnan') - mean(vals_sal, 'omitnan');
            sigMask(b,ch) = p < 0.05;
        end
    end
end

%% Significance matrix
catMat = zeros(size(pMat));
catMat(sigMask & deltaMat > 0) = 1;
catMat(sigMask & deltaMat < 0) = -1;

img = nan(size(catMat));
img(catMat == -1) = 1;
img(catMat == 0)  = 2;
img(catMat == 1)  = 3;

cmap = [
    0.75 0.85 1.00
    0.93 0.93 0.93
    1.00 0.80 0.85
    ];

fig500 = figure(500); clf;
set(fig500, 'Color', 'w', 'Position', [866 487 935 469]);

ax = subplot(2,3,1);
ax.Position = [0.1 0.4362 0.24 0.44];
imagesc(img);
colormap(cmap);
caxis([1 3]);

set(gca, 'XAxisLocation', 'top');
xticks(1:n_ch);
xticklabels(ch_names);
yticks(1:n_band);
yticklabels(b_names);

set(gca, ...
    'YDir', 'normal', ...
    'TickLength', [0 0], ...
    'LineWidth', 1.2, ...
    'FontSize', 10, ...
    'FontName', 'Arial');

cb = colorbar('Location', 'eastoutside', ...
    'Position', [0.3558 0.705 0.0142 0.1500], ...
    'Ticks', [1.33, 2, 2.66], ...
    'TickLabels', {'\downarrow oxy', 'n.s.', '\uparrow oxy'}, ...
    'LineWidth', 1, ...
    'FontSize', 10);

ax_pos = get(gca, 'Position');
set(gca, 'Position', [ax_pos(1) ax_pos(2) + 0.1 ax_pos(3) ax_pos(4) - 0.05]);

for b = 1:n_band
    for c = 1:n_ch
        if isnan(pMat(b,c))
            continue;
        end

        if pMat(b,c) < 1e-3
            txt = '<.001';
        else
            txt = sprintf('%.3f', pMat(b,c));
        end

        if pMat(b,c) < 0.05
            fw = 'normal';
            col = [0 0 0];
        else
            fw = 'normal';
            col = [0.4 0.4 0.4];
        end

        text(c, b, txt, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 10, ...
            'FontWeight', fw, ...
            'Color', col);
    end
end

%% GC plot
load('gc.mat');

inj_time = meta_info.inj_time;
drug = meta_info.drug;
skip_ratio = meta_info.skip_ratio;

isEmptyGC = cellfun(@isempty, gc(:,1));
valid_fileIdx = find((skip_ratio <= 10) & ~isEmptyGC);

drug = drug(valid_fileIdx);
inj_time = inj_time(valid_fileIdx,:);

gc_all = nan([512, 3, 2, length(valid_fileIdx), 3]);

for newIdx = 1:length(valid_fileIdx)
    fileIdx = valid_fileIdx(newIdx);
    gc_data = real(gc{fileIdx,1});
    t = gc{fileIdx,2};

    if inj_time(newIdx,1) > 0
        inj_start_end = hb_findIdx(inj_time(newIdx,:), t);

        for timeIdx = 1:3
            switch timeIdx
                case 1
                    time_win = inj_start_end(1) - 20 * 60 + 1 : inj_start_end(1);
                    time_win = time_win(time_win > 0);
                case 2
                    time_win = inj_start_end(end) : inj_start_end(end) + 20 * 60;
                case 3
                    time_win = inj_start_end(end) + 20 * 60 : length(t);
            end

            indiv_gc = nanmedian(gc_data(time_win,:,:,:), 1);
            gc_all(:,:,:,newIdx,timeIdx) = indiv_gc;
        end
    end
end

gc_lineplot_with_inset(gc_all, drug);

%% Driving synchrony and gating information
ch_mPFC = 1;
ch_BLA  = 2;
ch_A1   = 3;

n_sessions = 124;
duration_post = 20 * 60;
range_psd = [8 50];
range_plv = [4 8];
range_gc  = [4 8];

fprintf('Extracting data...\n');

data_mat = nan(n_sessions, 5); % [AC power, mPFC power, PLV, GC, drug]

for i = 1:n_sessions
    if i > size(psd,1) || isempty(psd{i,1}) || isempty(plv{i,1}) || isempty(gc{i,1}) || isnan(inj_time(i,2))
        continue;
    end

    t_start = inj_time(i, 2) + 60*5;
    t_end   = t_start + duration_post;

    idx_p = (psd{i,2} > t_start) & (psd{i,2} <= t_end);
    idx_l = (plv{i,2} > t_start) & (plv{i,2} <= t_end);
    idx_g = (gc{i,2} > t_start) & (gc{i,2} <= t_end);

    if ~any(idx_p) || ~any(idx_l) || ~any(idx_g)
        continue;
    end

    f_p = linspace(0, 121, size(psd{i,1}, 2));
    c_p = (f_p >= range_psd(1)) & (f_p <= range_psd(2));

    raw_p_ac = psd{i,1}(idx_p, c_p, ch_A1);
    val_psd_AC = 10 * log10(mean(raw_p_ac(:), 'omitnan'));

    raw_p_mpfc = psd{i,1}(idx_p, c_p, ch_mPFC);
    val_psd_mPFC = 10 * log10(mean(raw_p_mpfc(:), 'omitnan'));

    f_l = linspace(0, 100, size(plv{i,1}, 2));
    c_l = (f_l >= range_plv(1)) & (f_l <= range_plv(2));
    raw_l = plv{i,1}(idx_l, c_l, 2);
    val_plv = mean(raw_l(:), 'omitnan');

    f_g = linspace(0, 512, size(gc{i,1}, 2));
    c_g = (f_g >= range_gc(1)) & (f_g <= range_gc(2));
    raw_g = gc{i,1}(idx_g, c_g, 2, 2);
    val_gc = mean(raw_g(:), 'omitnan');

    data_mat(i, :) = [val_psd_AC, val_psd_mPFC, val_plv, val_gc, drug(i)];
end

idx_invalid = any(~isfinite(data_mat), 2);
idx_gc_outlier = (data_mat(:,4) <= 1e-5) | (data_mat(:,4) > 0.3);
idx_low_power = (data_mat(:,1) < -20) | (data_mat(:,2) < -20);

valid_mask = ~idx_invalid & ~idx_gc_outlier & ~idx_low_power;
clean_data = data_mat(valid_mask, :);

fprintf('--- Filtering summary ---\n');
fprintf('Total sessions: %d -> Final sessions: %d\n', n_sessions, size(clean_data, 1));
fprintf('Excluded: invalid (%d), GC outlier (%d), low power (%d)\n', ...
    sum(idx_invalid), ...
    sum(idx_gc_outlier & ~idx_invalid), ...
    sum(idx_low_power & ~idx_invalid & ~idx_gc_outlier));

%% Correlation plots
plot_configs = {
    'Driving Sync (AC)',   1, 3, 'AC Power (dB)',   'AC-mPFC \theta_{low} PLV', 5
    'Driving Sync (mPFC)', 2, 3, 'mPFC Power (dB)', 'AC-mPFC \theta_{low} PLV', 6
    'Gating Info (AC)',    1, 4, 'AC Power (dB)',   'AC\rightarrowmPFC \theta_{low} GC', 7
    'Gating Info (mPFC)',  2, 4, 'mPFC Power (dB)', 'AC\rightarrowmPFC \theta_{low} GC', 8
    };

g_oxy = clean_data(clean_data(:,5) == 1, :);
g_sal = clean_data(clean_data(:,5) == 0, :);

colors = {[0.85 0.2 0.2], [0.3 0.3 0.3]};
alphas = [0.4, 0.3];
std_font_sz = 10;

for i = 1:4
    target_pos = plot_configs{i, 6};

    subplot(2, 4, target_pos);
    cla;
    axis square;
    hold on;

    x_idx = plot_configs{i, 2};
    y_idx = plot_configs{i, 3};

    groups = {g_oxy, g_sal};
    r_vals = [NaN, NaN];
    p_vals = [NaN, NaN];

    xlim([1.5 5.5]);
    ax_limit = xlim;

    for g = 1:2
        curr_data = groups{g};
        scatter(curr_data(:, x_idx), curr_data(:, y_idx), 20, colors{g}, ...
            'filled', 'MarkerFaceAlpha', alphas(g), 'HandleVisibility', 'off');
    end

    for g = 1:2
        curr_data = groups{g};
        x_val = curr_data(:, x_idx);
        y_val = curr_data(:, y_idx);

        if length(x_val) > 2
            [r, p] = corr(x_val, y_val, 'Rows', 'complete');
            r_vals(g) = r;
            p_vals(g) = p;

            l_style = ':';
            l_width = 1.2;
            if p < 0.05
                l_style = '-';
                l_width = 1.8;
            end

            p_fit = polyfit(x_val, y_val, 1);
            x_range = linspace(ax_limit(1), ax_limit(2), 100);

            plot(x_range, polyval(p_fit, x_range), ...
                'Color', colors{g}, ...
                'LineStyle', l_style, ...
                'LineWidth', l_width, ...
                'HandleVisibility', 'off');
        end
    end

    set(gca, ...
        'Box', 'off', ...
        'TickDir', 'in', ...
        'LineWidth', 1, ...
        'FontSize', std_font_sz);

    set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');

    xlabel(plot_configs{i, 4}, 'FontSize', std_font_sz);
    ylabel(plot_configs{i, 5}, 'FontSize', std_font_sz);

    text(0.95, 0.94, sprintf('oxy: r=%.2f, p=%.2f', r_vals(1), p_vals(1)), ...
        'Units', 'normalized', ...
        'Color', colors{1}, ...
        'FontSize', 10, ...
        'HorizontalAlignment', 'right', ...
        'FontWeight', 'normal');

    text(0.95, 0.86, sprintf('sal: r=%.2f, p=%.2f', r_vals(2), p_vals(2)), ...
        'Units', 'normalized', ...
        'Color', colors{2}, ...
        'FontSize', 10, ...
        'HorizontalAlignment', 'right', ...
        'FontWeight', 'normal');
end

%% Local function
function gc_lineplot_with_inset(gc_all, drug)

freq = 1:55;
theta_range = [8 12];

chanLabel = {'mPFC', 'BLA', 'AC'};
chanComb = [1 2; 1 3; 2 3];
timeIdx = 2;

sal_outliers = [4, 13, 14, 16, 18, 19];
oxy_outliers = [2, 16];

sal_idx = find(drug == 0);
sal_idx(sal_outliers) = [];

oxy_idx = find(drug == 1);
oxy_idx(oxy_outliers) = [];

for combIdx = 2
    for dirIdx = 1:2

        sub_ax = subplot(2,3,-dirIdx+4);
        hold on;
        axis square;
        cla;
        hold on;

        gc_sal_raw = squeeze(gc_all(:, combIdx, dirIdx, sal_idx, timeIdx))';
        gc_oxy_raw = squeeze(gc_all(:, combIdx, dirIdx, oxy_idx, timeIdx))';

        valid_sal = all(~isnan(gc_sal_raw), 2) & any(gc_sal_raw ~= 0, 2);
        valid_oxy = all(~isnan(gc_oxy_raw), 2) & any(gc_oxy_raw ~= 0, 2);

        gc_sal = gc_sal_raw(valid_sal, freq);
        gc_oxy = gc_oxy_raw(valid_oxy, freq);

        mean_sal = nanmean(gc_sal, 1);
        sem_sal = nanstd(gc_sal, 0, 1) / sqrt(size(gc_sal,1));
        mean_oxy = nanmean(gc_oxy, 1);
        sem_oxy = nanstd(gc_oxy, 0, 1) / sqrt(size(gc_oxy,1));

        ms_sal = smoothdata(mean_sal, 'movmean', 9);
        ss_sal = smoothdata(sem_sal, 'movmean', 5);
        ms_oxy = smoothdata(mean_oxy, 'movmean', 9);
        ss_oxy = smoothdata(sem_oxy, 'movmean', 5);

        fill([freq, fliplr(freq)], [ms_sal + ss_sal, fliplr(ms_sal - ss_sal)], ...
            [0.6 0.6 0.6], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(freq, ms_sal, 'Color', [0.3 0.3 0.3], 'LineWidth', 1);

        fill([freq, fliplr(freq)], [ms_oxy + ss_oxy, fliplr(ms_oxy - ss_oxy)], ...
            [1 0.3 0.3], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(freq, ms_oxy, 'Color', [0.85 0.2 0.2], 'LineWidth', 1);

        for f = 1:length(freq)
            [~, p] = ttest2(gc_sal(:,f), gc_oxy(:,f));
            if p < 0.05
                col = 'k';
                if ms_oxy(f) > ms_sal(f)
                    col = 'r';
                end
                text(freq(f), 0.23, '*', ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 10, ...
                    'Color', col, ...
                    'FontWeight', 'normal');
            end
        end

        xlim([1 55]);
        xticks([10 20 30 40 50]);
        ylim([0.05 0.25]);
        yticks([0.05 0.10 0.15 0.20 0.25]);

        set(gca, ...
            'Box', 'off', ...
            'TickDir', 'in', ...
            'LineWidth', 1, ...
            'FontSize', 10, ...
            'FontName', 'Arial', ...
            'FontWeight', 'normal');

        from_ch = chanLabel{chanComb(combIdx, dirIdx)};
        to_ch = chanLabel{chanComb(combIdx, 3 - dirIdx)};

        title(sprintf('%s \\rightarrow %s', from_ch, to_ch), ...
            'FontWeight', 'normal', ...
            'FontSize', 10);

        ylabel('GC (a.u.)', 'FontWeight', 'normal');
        xlabel('Freq (Hz)', 'FontWeight', 'normal');

        pos = get(sub_ax, 'Position');
        inset_pos = [pos(1) + pos(3) * 0.55, pos(2) + pos(4) * 0.35, pos(3) * 0.35, pos(4) * 0.35];

        axes('Position', inset_pos);
        hold on;

        f_idx = freq >= theta_range(1) & freq <= theta_range(2);
        val_sal = nanmean(gc_sal(:, f_idx), 2);
        val_oxy = nanmean(gc_oxy(:, f_idx), 2);

        m_bars = [mean(val_oxy), mean(val_sal)];
        s_bars = [std(val_oxy) / sqrt(length(val_oxy)), std(val_sal) / sqrt(length(val_sal))];

        b = bar(1:2, m_bars, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6);
        b.CData(2,:) = [0.7 0.7 0.7];
        b.CData(1,:) = [0.85 0.2 0.2];

        errorbar(1:2, m_bars, s_bars, 'k.', 'LineWidth', 1, 'CapSize', 6);

        jitter = 0.15;
        scatter(ones(size(val_oxy)) + (rand(size(val_oxy)) - 0.5) * jitter, val_oxy, ...
            8, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
        scatter(2 * ones(size(val_sal)) + (rand(size(val_sal)) - 0.5) * jitter, val_sal, ...
            8, 'k', 'filled', 'MarkerFaceAlpha', 0.3);

        [~, p_bar] = ttest2(val_oxy, val_sal);

        max_val = max([val_oxy; val_sal]);
        if max_val <= 0
            max_val = 0.1;
        end

        y_sig_line = max_val * 1.15;
        line([1 2], [y_sig_line y_sig_line], 'Color', 'k', 'LineWidth', 1);

        sig_txt = 'n.s.';
        if p_bar < 0.001
            sig_txt = '***';
        elseif p_bar < 0.01
            sig_txt = '**';
        elseif p_bar < 0.05
            sig_txt = '*';
        end

        if p_bar < 0.05
            text(1.5, y_sig_line, sig_txt, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 14, ...
                'FontWeight', 'normal');
        else
            text(1.5, y_sig_line, sig_txt, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 9, ...
                'FontWeight', 'normal');
        end

        set(gca, ...
            'Color', 'none', ...
            'Box', 'off', ...
            'XTick', [1 2], ...
            'XTickLabel', {'oxy', 'sal'}, ...
            'FontSize', 10, ...
            'FontWeight', 'normal', ...
            'LineWidth', 1);

        xlim([0.3 2.7]);
        ylim([0, max_val * 1.5]);
        title('\theta_{low}', 'FontSize', 10, 'FontWeight', 'normal');
    end
end
end
