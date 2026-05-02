%% ============================================================
% 探查脚本 v2：基于傅里叶分析的平稳波动信号去噪探索
% 适用于：围绕中心值上下波动的位移类信号
% 数据文件：Attachment 3.xlsx（训练集 Sheet）
% ============================================================
clear; clc; close all;

%% ---- 1. 读取并准备数据 ----
filename = 'ap4.xlsx';
sheetName = '训练集';
opts = detectImportOptions(filename, 'Sheet', sheetName);
opts.VariableNamingRule = 'preserve';
dataTable = readtable(filename, opts, 'Sheet', sheetName);

N = height(dataTable);
serialNo = (1:N)';                     % 序列序号：1,2,3,...
rawDisplacement = dataTable{:, 2};   % 表面位移（第2列）

N = length(rawDisplacement);

% ---- 基本插值（处理零值与NaN）----
x = rawDisplacement;
x(x == 0) = NaN;  % 将0统一视为缺失（可调整）
validMask = ~isnan(x);
x_interp = interp1(find(validMask), x(validMask), 1:N, 'linear', 'extrap');
% 确保列向量
x_interp = x_interp(:);

%% ---- 2. 频谱探查（决定后续滤波器参数） ----
% 去均值以消除直流分量对频谱的污染
x_detrend = x_interp - mean(x_interp);

% Welch法功率谱估计（比直接FFT更平滑）
[Pxx, f] = pwelch(x_detrend, hamming(256), 128, 1024, 1);  
% 假设采样间隔为1（时间序号），频率单位：周期/序号

figure('Position', [100, 100, 1000, 500]);
semilogy(f, Pxx, 'b-', 'LineWidth', 1.5);
xlabel('频率 (周期/序号)'); ylabel('功率谱密度');
title('功率谱密度估计 —— 平稳波动信号的能量分布');
grid on;

% 自动标注能量最大的前三个频率峰（帮助你定位周期成分）
[~, locs] = findpeaks(Pxx, f, 'SortStr', 'descend', 'NPeaks', 3);
hold on;
plot(locs, interp1(f, Pxx, locs), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
text(locs, interp1(f, Pxx, locs), arrayfun(@(x) sprintf('%.3f', x), locs, ...
    'UniformOutput', false), 'VerticalAlignment', 'bottom');
legend('功率谱密度', '主要峰值频率');

%% ---- 3. 基于频谱观察的低通/带通滤波器设计 ----
% 
% 请根据上一步绘制的功率谱图，在这里手工设定截止频率。
% 例：若主要周期成分频率 < 0.1，噪声在 >0.2，
% 则可设置低通截止频率 fc = 0.15
%
% 这里提供三种可选方案，请取消注释你需要的那一种：
% --------------------------------------------------------

% ----- 方案A：简单低通（默认） -----
fc = 0.05;          % 截止频率（根据频谱修改）
[b, a] = butter(6, fc/(0.5), 'low');
x_filtered = filtfilt(b, a, x_interp);

% ----- 方案B：带通滤波（若已知有用频带） -----
% f_low = 0.02; f_high = 0.15;
% [b, a] = butter(4, [f_low f_high]/(0.5), 'bandpass');
% x_filtered = filtfilt(b, a, x_interp);

% ----- 方案C：基于阈值的小波重建（备选） -----
% 使用 wdenoise 的效果也可以一并对比
x_wavelet = wdenoise(x_interp(:));
x_wavelet = x_wavelet(:);

%% ---- 4. 时域效果对比 ----
figure('Position', [100, 100, 1400, 900]);

subplot(3,1,1);
plot(serialNo, x_interp, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8); hold on;
plot(serialNo, x_filtered, 'r-', 'LineWidth', 1.2);
title('低通/带通频域滤波效果');
xlabel('Serial No.'); ylabel('位移 (mm)');
legend('插值后', '频域滤波'); grid on;

subplot(3,1,2);
plot(serialNo, x_interp, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8); hold on;
plot(serialNo, x_wavelet, 'b-', 'LineWidth', 1.2);
title('小波去噪对比');
xlabel('Serial No.'); ylabel('位移 (mm)');
legend('插值后', '小波去噪'); grid on;

subplot(3,1,3);
plot(serialNo, rawDisplacement, 'k-', 'LineWidth', 0.5); hold on;
plot(serialNo, x_filtered, 'r-', 'LineWidth', 1.5);
title('原始数据 vs 频域滤波结果');
xlabel('Serial No.'); ylabel('位移 (mm)');
legend('原始', '频域滤波'); grid on;

sgtitle('基于傅里叶分析的平稳信号去噪探查');

%% ---- 5. 滤波残差与信噪比 ----
res = x_interp - x_filtered;
snr_freq = 20 * log10(std(x_interp) / std(res));
fprintf('频域滤波 SNR = %.2f dB\n', snr_freq);

% 若同时运行了小波去噪，也可计算小波SNR
res_w = x_interp - x_wavelet;
snr_wav = 20 * log10(std(x_interp) / std(res_w));
fprintf('小波去噪 SNR = %.2f dB\n', snr_wav);