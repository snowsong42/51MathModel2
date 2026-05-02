%% ============================================================
% 降噪效果检查脚本：频谱对比 + 信噪比输出
% 读取 Filtered_Result.csv，含三列：SerialNo, RawDisplacement, FilteredDisplacement
% ============================================================
clear; clc; close all;

%% ---- 1. 读取数据 ----
filename = 'Filtered_Result.csv';
data = readtable(filename);
raw = data.RawDisplacement;
filtered = data.FilteredDisplacement;

% 去除缺失值（以防万一）
valid = ~isnan(raw) & ~isnan(filtered);
raw = raw(valid);
filtered = filtered(valid);
N = length(raw);
fprintf('有效数据点数：%d\n', N);

%% ---- 2. 计算信噪比 ----
residual = raw - filtered;
SNR = 20 * log10(std(raw) / std(residual));
fprintf('信噪比 SNR = %.2f dB\n', SNR);

%% ---- 3. 功率谱估计与绘图 ----
% 设定 Welch 谱估计参数
window = min(256, N);
noverlap = round(window/2);
nfft = 1024;

% 降噪前功率谱
[Pxx_raw, f] = pwelch(raw - mean(raw), hamming(window), noverlap, nfft, 1);

% 降噪后功率谱
[Pxx_filt, ~] = pwelch(filtered - mean(filtered), hamming(window), noverlap, nfft, 1);

% 绘制降噪前频谱
figure('Position', [100, 100, 800, 500]);
semilogy(f, Pxx_raw, 'b-', 'LineWidth', 1.5);
xlabel('频率 (周期/序号)'); ylabel('功率谱密度');
title(sprintf('降噪前功率谱 (SNR = %.2f dB)', SNR));
grid on;
%saveas(gcf, 'before_denoise_psd.png');
%fprintf('降噪前频谱图已保存至 before_denoise_psd.png\n');

% 绘制降噪后频谱
figure('Position', [100, 100, 800, 500]);
semilogy(f, Pxx_filt, 'r-', 'LineWidth', 1.5);
xlabel('频率 (周期/序号)'); ylabel('功率谱密度');
title(sprintf('降噪后功率谱 (SNR = %.2f dB)', SNR));
grid on;
%saveas(gcf, 'after_denoise_psd.png');
%fprintf('降噪后频谱图已保存至 after_denoise_psd.png\n');

% 可选：叠加对比图
figure('Position', [100, 100, 800, 500]);
semilogy(f, Pxx_raw, 'b-', 'LineWidth', 1.2); hold on;
semilogy(f, Pxx_filt, 'r-', 'LineWidth', 1.2);
xlabel('频率 (周期/序号)'); ylabel('功率谱密度');
title(sprintf('频谱对比 (SNR = %.2f dB)', SNR));
legend('降噪前', '降噪后');
grid on;
%saveas(gcf, 'spectrum_comparison.png');
%fprintf('频谱对比图已保存至 spectrum_comparison.png\n');

disp('全部检查完成。');