%% ============================================================
% 降噪效果检查脚本：频谱对比 + 信噪比输出 (含包络包裹填色)
% 修正版：修复 MinPeakDistance 错误
% ============================================================
clear; clc; close all;

%% ---- 1. 读取数据 ----
filename = 'Filtered_Result.csv';
data = readtable(filename);
raw = data.RawDisplacement;
filtered = data.FilteredDisplacement;

valid = ~isnan(raw) & ~isnan(filtered);
raw = raw(valid);
filtered = filtered(valid);
N = length(raw);
fprintf('有效数据点数：%d\n', N);

%% ---- 2. 计算信噪比 ----
residual = raw - filtered;
SNR = 20 * log10(std(raw) / std(residual));
fprintf('信噪比 SNR = %.2f dB\n', SNR);

%% ---- 3. 功率谱估计 ----
window = min(256, N);
noverlap = round(window/2);
nfft = 1024;

[Pxx_raw, f] = pwelch(raw - mean(raw), hamming(window), noverlap, nfft, 1);
[Pxx_filt, ~] = pwelch(filtered - mean(filtered), hamming(window), noverlap, nfft, 1);

%% ---- 4. 计算上下包络线（修正后的函数） ----
[f_upper_raw, P_upper_raw] = getEnvelope(f, Pxx_raw, 'upper');
[f_lower_raw, P_lower_raw] = getEnvelope(f, Pxx_raw, 'lower');

[f_upper_filt, P_upper_filt] = getEnvelope(f, Pxx_filt, 'upper');
[f_lower_filt, P_lower_filt] = getEnvelope(f, Pxx_filt, 'lower');

%% ---- 5. 绘图 ----
figure('Position', [100, 100, 1000, 650]);
ax = gca;
set(ax, 'YScale', 'log');
hold on;

% 降噪后填充
fill([f_upper_filt; flipud(f_lower_filt)], ...
     [P_upper_filt; flipud(P_lower_filt)], ...
     'r', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
plot(f_upper_filt, P_upper_filt, 'r--', 'LineWidth', 1.0);
plot(f_lower_filt, P_lower_filt, 'r--', 'LineWidth', 1.0);
plot(f, Pxx_filt, 'r-', 'LineWidth', 1.2);

% 降噪前填充
fill([f_upper_raw; flipud(f_lower_raw)], ...
     [P_upper_raw; flipud(P_lower_raw)], ...
     'b', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
plot(f_upper_raw, P_upper_raw, 'b--', 'LineWidth', 1.0);
plot(f_lower_raw, P_lower_raw, 'b--', 'LineWidth', 1.0);
plot(f, Pxx_raw, 'b-', 'LineWidth', 1.2);


xlabel('频率 (周期/序号)');
ylabel('功率谱密度');
title(sprintf('降噪前后傅里叶频谱对比 (信噪比 = %.2f dB)', SNR));
legend('降噪后包络填充','降噪后上包络','降噪后下包络','降噪后功率谱',...
        '降噪前包络填充','降噪前上包络','降噪前下包络','降噪前功率谱',...
       'Location','best');
grid on;

%% ============================================================
% 辅助函数：提取频谱的上下包络（修正版）
% ============================================================
function [f_env, env_vals] = getEnvelope(f, Pxx, mode)
    % 按样本索引找峰值，避免 MinPeakDistance 单位问题
    minPeakDistance = max(3, round(length(f)/50));  % 样本点间隔
    if strcmp(mode, 'upper')
        [pks, idx] = findpeaks(Pxx, 'MinPeakDistance', minPeakDistance);
    else
        [pks, idx] = findpeaks(-Pxx, 'MinPeakDistance', minPeakDistance);
        pks = -pks;
    end
    locs = f(idx);
    % 插值到原频率网格
    if length(locs) >= 2
        env_vals = interp1(locs, pks, f, 'pchip', 'extrap');
    else
        env_vals = ones(size(f)) * mean(Pxx);
    end
    f_env = f;
end