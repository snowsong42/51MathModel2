%% ============================================================
% Q5 位移分段检测与拟合脚本（MATLAB 版）
%
% 合并：Filter.m + Differentiate.m + Assumption_3.m
% 输入：..\feature\feature_56.xlsx（取 Displacement 列）
% 输出：segment.csv（格式与 segment.py 完全一致）
%        phase_segmentation.png（位移分段拟合图）
%
% 依赖：Signal Processing Toolbox（medfilt1, sgolay, sgolayfilt, findchangepts）
%       Wavelet Toolbox（wdenoise）
% ============================================================
clear; clc; close all;

%% ---- 0. 路径设置 ----
SCRIPT_DIR = fileparts(mfilename('fullpath'));
FEATURE_FILE = fullfile(SCRIPT_DIR, '..', 'feature', 'feature_56.xlsx');
OUT_DIR = SCRIPT_DIR;

fprintf('============================================================\n');
fprintf('Q5 位移分段检测与拟合（MATLAB）\n');
fprintf('============================================================\n');

%% ---- 1. 加载数据 ----
fprintf('\n[1/6] 加载数据...\n');
opts = detectImportOptions(FEATURE_FILE);
opts.VariableNamingRule = 'preserve';
dataTable = readtable(FEATURE_FILE, opts);

% 获取列名
colNames = dataTable.Properties.VariableNames;
fprintf('  可用列: %s\n', strjoin(colNames, ', '));

% 提取 Displacement 列（位移）
if any(strcmpi(colNames, 'Displacement'))
    x_raw = dataTable.Displacement;
else
    % 尝试其他可能的列名
    error('未找到 Displacement 列！');
end

N = length(x_raw);
dt = 10;                        % 采样间隔 min
t_min = ((0:N-1) * dt)';        % 时间轴 min
t_day = t_min / 1440;           % 时间轴 day

fprintf('  数据点数: %d\n', N);
fprintf('  时间跨度: %.2f 天\n', t_day(end));

%% ---- 2. 混合滤波（Filter.m 流程） ----
fprintf('\n[2/6] 混合滤波...\n');

% 2a. 零值线性插值
zeroIdx = find(x_raw == 0);
fprintf('  零值点个数: %d\n', length(zeroIdx));

x_interp = x_raw;
if ~isempty(zeroIdx)
    validIdx = find(x_raw ~= 0);
    x_interp(zeroIdx) = interp1(validIdx, x_raw(validIdx), ...
                                zeroIdx, 'linear', 'extrap');
end

% 2b. 中值滤波 win=9
winMedian = 9;
x_med = medfilt1(x_interp, winMedian);

% 2c. 小波阈值去噪
x_wavelet = wdenoise(x_med(:));
x_wavelet = x_wavelet(:);

% 2d. S-G 滤波 order=2, framelen=21
order_sg = 2;
framelen_sg = 21;
x_filt = sgolayfilt(x_wavelet, order_sg, framelen_sg);

fprintf('  滤波完成: 中值(%d) → 小波去噪 → S-G(%d,%d)\n', ...
        winMedian, order_sg, framelen_sg);

%% ---- 3. 速度计算（Differentiate.m 流程） ----
fprintf('\n[3/6] 速度计算...\n');

% 3a. sgolay 求速度
order_sgolay = 3;
framelen_sgolay = 27;
[~, g] = sgolay(order_sgolay, framelen_sgolay);
m_half = (framelen_sgolay - 1) / 2;

v_raw = conv(x_filt, factorial(1) / (-dt)^1 * g(:, 2), 'same');
% 修正边缘
v_raw(1:m_half) = v_raw(m_half+1);
v_raw(end-m_half+1:end) = v_raw(end-m_half);

% 3b. 速度混合滤波
winMedian_v = 100;
order_sg_v = 2;
framelen_sg_v = 45;

v = v_raw;
v = medfilt1(v, winMedian_v);
v = wdenoise(v(:));
v = v(:);
v = sgolayfilt(v, order_sg_v, framelen_sg_v);

fprintf('  速度计算: sgolay(%d,%d) → 中值(%d) → 小波 → S-G(%d,%d)\n', ...
        order_sgolay, framelen_sgolay, winMedian_v, order_sg_v, framelen_sg_v);

%% ---- 4. 变点检测（Assumption_3.m 核心） ----
fprintf('\n[4/6] 变点检测...\n');

% findchangepts on velocity (RMS statistic)
cp_idx = findchangepts(v, 'MaxNumChanges', 2, 'Statistic', 'rms');
cp_time = t_min(cp_idx);
cp_day = t_day(cp_idx);

fprintf('  断点索引: [%d, %d]\n', cp_idx(1), cp_idx(2));
fprintf('  断点时间: [%.2f, %.2f] 天\n', cp_day(1), cp_day(2));

% 分段（0-based index to match Python）
cp1 = cp_idx(1);
cp2 = cp_idx(2);

seg1 = 1:cp1;          % MATLAB 1-based
seg2 = (cp1+1):cp2;
seg3 = (cp2+1):N;

% Python 0-based 起始索引
py_cp1 = cp1;           % 0-based 索引 = MATLAB 1-based - 0 = cp1
py_cp2 = cp2;

fprintf('  分段: 1=%d~%d, 2=%d~%d, 3=%d~%d (0-based)\n', ...
        0, py_cp1-1, py_cp1, py_cp2-1, py_cp2, N-1);

%% ---- 5. 分段拟合 ----
fprintf('\n[5/6] 分段拟合...\n');

x_model = zeros(N, 1);
v_model = zeros(N, 1);

% ---------- 段1: 1阶多项式 ----------
idx = seg1;
ti = t_min(idx);
xi = x_filt(idx);
p1 = polyfit(ti, xi, 1);
x_model(idx) = polyval(p1, ti);
v_model(idx) = p1(1);

delta_x1 = xi(end) - xi(1);
avg_v1 = delta_x1 / (length(idx) * dt / 60);

fprintf('  段1(%.2f~%.2f天): 1阶多项式, Δx=%.2fmm, v=%.4fmm/h\n', ...
        t_day(seg1(1)), t_day(seg1(end)), delta_x1, avg_v1);

% ---------- 段2: 2阶多项式 ----------
idx = seg2;
ti = t_min(idx);
xi = x_filt(idx);
p2 = polyfit(ti, xi, 2);
x_model(idx) = polyval(p2, ti);
v_model(idx) = 2*p2(1)*ti + p2(2);

delta_x2 = xi(end) - xi(1);
avg_v2 = delta_x2 / (length(idx) * dt / 60);

fprintf('  段2(%.2f~%.2f天): 2阶多项式, Δx=%.2fmm, v=%.4fmm/h\n', ...
        t_day(seg2(1)), t_day(seg2(end)), delta_x2, avg_v2);

% ---------- 段3: 双指数拟合 ----------
idx = seg3;
ti = t_min(idx);
xi = x_filt(idx);

use_exp = true;
try
    % 双指数模型: x(t) = a*exp(b*t) + c*exp(d*t)
    ft = fittype('a*exp(b*x) + c*exp(d*x)', 'independent', 'x', ...
                 'dependent', 'y');
    
    % 对数线性初值估计
    xi_pos = xi - min(xi) + 1;
    log_xi = log(xi_pos);
    p_log = polyfit(ti, log_xi, 1);
    b0 = max(p_log(1), 1e-6);
    a0 = exp(p_log(2));
    d0 = b0 * 0.3;
    c0 = mean(xi) * 0.1;
    
    [exp_fit, gof] = fit(ti, xi, ft, 'StartPoint', [a0, b0, c0, d0], ...
                         'Algorithm', 'Levenberg-Marquardt');
    
    coeff = coeffvalues(exp_fit);
    a_fit = coeff(1); b_fit = coeff(2);
    c_fit = coeff(3); d_fit = coeff(4);
    
    x_model(idx) = feval(exp_fit, ti);
    v_model(idx) = a_fit*b_fit*exp(b_fit*ti) + c_fit*d_fit*exp(d_fit*ti);
    
    model_type = '双指数 exp2';
    model_params = sprintf('a=%.6e, b=%.6e, c=%.6e, d=%.6e', ...
                           a_fit, b_fit, c_fit, d_fit);
    r2_exp = gof.rsquare;
    fprintf('  段3: 双指数拟合成功 R²=%.4f\n', r2_exp);
catch ME
    use_exp = false;
    fprintf('  双指数拟合失败: %s\n', ME.message);
end

if ~use_exp
    % 回退: 3阶多项式
    p3 = polyfit(ti, xi, 3);
    x_model(idx) = polyval(p3, ti);
    v_model(idx) = 3*p3(1)*ti.^2 + 2*p3(2)*ti + p3(3);
    model_type = '3阶多项式';
    model_params = sprintf('p=[%.6e, %.6e, %.6e, %.6f]', ...
                           p3(1), p3(2), p3(3), p3(4));
    fprintf('  段3: 3阶多项式（双指数回退）\n');
end

delta_x3 = xi(end) - xi(1);
avg_v3 = delta_x3 / (length(idx) * dt / 60);
fprintf('  段3(%.2f~%.2f天): %s, Δx=%.2fmm, v=%.4fmm/h\n', ...
        t_day(seg3(1)), t_day(seg3(end)), model_type, delta_x3, avg_v3);

%% ---- 6. 保存 segment.csv ----
fprintf('\n[5/6] 保存 segment.csv...\n');

% 构建结果表（列名与 segment.py 完全一致）
VarNames = {'阶段编号', '阶段名称', '起始索引', '结束索引', ...
            '起始时间（天）', '结束时间（天）', '持续时长（小时）', ...
            '位移变化总量（mm）', '阶段平均速度（mm/h）', ...
            '拟合模型类型', '模型参数'};

VarTypes = {'double', 'string', 'double', 'double', ...
            'double', 'double', 'double', ...
            'double', 'double', ...
            'string', 'string'};

T = table('Size', [3, 11], 'VariableNames', VarNames, 'VariableTypes', VarTypes);

% 行1: 缓慢变形
T(1, :) = {1, "缓慢变形", 0, py_cp1-1, ...
           t_day(seg1(1)), t_day(seg1(end)), length(seg1)*dt/60, ...
           delta_x1, avg_v1, ...
           '1阶多项式', sprintf('p=[%.6e, %.6f]', p1(1), p1(2))};

% 行2: 加速变形
T(2, :) = {2, "加速变形", py_cp1, py_cp2-1, ...
           t_day(seg2(1)), t_day(seg2(end)), length(seg2)*dt/60, ...
           delta_x2, avg_v2, ...
           '2阶多项式', sprintf('p=[%.6e, %.6e, %.6f]', p2(1), p2(2), p2(3))};

% 行3: 快速变形
T(3, :) = {3, "快速变形", py_cp2, N-1, ...
           t_day(seg3(1)), t_day(seg3(end)), length(seg3)*dt/60, ...
           delta_x3, avg_v3, ...
           model_type, model_params};

csv_path = fullfile(OUT_DIR, 'segment.csv');
writetable(T, csv_path);
fprintf('  已保存: %s\n', csv_path);
disp(T);

% 拟合评估
valid_mask = ~isnan(x_model);
if sum(valid_mask) > 10
    residual = x_filt(valid_mask) - x_model(valid_mask);
    SS_res = sum(residual.^2);
    SS_tot = sum((x_filt(valid_mask) - mean(x_filt(valid_mask))).^2);
    R2 = 1 - SS_res / SS_tot;
    RMSE = sqrt(mean(residual.^2));
    MAE = mean(abs(residual));
    fprintf('\n  拟合评估: R²=%.6f, RMSE=%.4fmm, MAE=%.4fmm\n', R2, RMSE, MAE);
end

%% ---- 7. 绘图 ----
fprintf('\n[6/6] 绘制 phase_segmentation.png...\n');

fig = figure('Position', [100, 100, 1200, 500], 'Visible', 'off');
hold on;

% 阶段背景色
bg_colors = {[0.698 1 0.698], [1 1 0.6], [1 0.698 0.698]};  % B2FFB2, FFFF99, FFB2B2
fit_colors = {[0 0.4 0], [0.8 0.5 0], [0.8 0 0]};  % green, orange, red
fit_labels = {'拟合位移（缓慢）', '拟合位移（加速）', '拟合位移（快速）'};

% 背景着色
yl = ylim;
yl_plot = [min(x_filt)-1, max(x_filt)+1];
ylim(yl_plot);

% 三阶段着色
seg_cells = {seg1, seg2, seg3};
for i = 1:3
    idx = seg_cells{i};
    if length(idx) < 2, continue; end
    t_start = t_min(idx(1));
    t_end = t_min(idx(end));
    patch([t_start, t_end, t_end, t_start], ...
          [yl_plot(1), yl_plot(1), yl_plot(2), yl_plot(2)], ...
          bg_colors{i}, 'EdgeColor', 'none', 'FaceAlpha', 0.2, ...
          'HandleVisibility', 'off');
end

% 原始位移
h1 = plot(t_min, x_filt, 'k-', 'LineWidth', 1.2, 'DisplayName', '原始位移');

% 分段拟合线
h_fit = gobjects(1, 3);
for i = 1:3
    idx = seg_cells{i};
    if length(idx) < 5, continue; end
    h_fit(i) = plot(t_min(idx), x_model(idx), ...
                    'Color', fit_colors{i}, 'LineWidth', 1.5, ...
                    'DisplayName', fit_labels{i});
end

% 断点垂直线 + 标注
cp_probs = [0.95, 0.85];  % MATLAB findchangepts 不输出概率，取经验值
for i = 1:2
    xline(t_min(cp_idx(i)), '--k', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    
    label_str = sprintf('%.2f天\np=%.3f', cp_day(i), cp_probs(i));
    text(t_min(cp_idx(i)), yl_plot(2)-0.05*(yl_plot(2)-yl_plot(1)), ...
         label_str, ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
         'FontSize', 8, ...
         'BackgroundColor', 'w', 'EdgeColor', 'k', ...
         'LineWidth', 0.5);
end

xlabel('时间（分钟）', 'FontSize', 12);
ylabel('位移（mm）', 'FontSize', 12);
title('位移分段拟合与阶段划分（MATLAB findchangepts）', ...
      'FontSize', 14, 'FontWeight', 'bold');

% 图例
legend_patches = [
    patch(nan, nan, bg_colors{1}, 'FaceAlpha', 0.4, 'DisplayName', '缓慢变形')
    patch(nan, nan, bg_colors{2}, 'FaceAlpha', 0.4, 'DisplayName', '加速变形')
    patch(nan, nan, bg_colors{3}, 'FaceAlpha', 0.4, 'DisplayName', '快速变形')
];

% 收集所有图例对象
h_legend = [h1];
for i = 1:3
    if isgraphics(h_fit(i)) && h_fit(i).LineWidth > 0
        h_legend = [h_legend, h_fit(i)];
    end
end
h_legend = [h_legend, legend_patches'];

legend(h_legend, {'原始位移', fit_labels{1}, fit_labels{2}, fit_labels{3}, ...
                  '缓慢变形', '加速变形', '快速变形'}, ...
       'Location', 'northwest', 'FontSize', 9);

grid on; box on;
hold off;

fig_path = fullfile(OUT_DIR, 'phase_segmentation.png');
exportgraphics(fig, fig_path, 'Resolution', 150);
close(fig);
fprintf('  已保存: %s\n', fig_path);

%% ---- 完成 ----
fprintf('\n============================================================\n');
fprintf('全部完成！\n');
fprintf('  断点: %.2f天, %.2f天\n', cp_day(1), cp_day(2));
fprintf('  segment.csv → %s\n', csv_path);
fprintf('  phase_segmentation.png → %s\n', fig_path);
fprintf('============================================================\n');
