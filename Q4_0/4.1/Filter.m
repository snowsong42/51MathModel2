%% ============================================================
% 混合滤波脚本：插值 → 中值滤波 → 小波阈值去噪 → S‑G 滤波
% 适用数据：含噪声/瞬时跳变的位移信号（含稀疏零值）
% 数据文件：Attachment 2.xlsx
% 需要：Wavelet Toolbox
% 提示：请将此脚本重命名为 hybrid_filter.m，避免与内置 filter 函数冲突
% ============================================================
clear; clc; close all;



%% ---- 1. 读取 Excel 数据（保留原始列名，消除警告） ----
filename = 'Attachment 4.xlsx';
sheetName = '训练集';                    % 读取训练集 Sheet
opts = detectImportOptions(filename, 'Sheet', sheetName);
opts.VariableNamingRule = 'preserve';   % 关键：保留原始列名，不修改
dataTable = readtable(filename, opts, 'Sheet', sheetName);
% 根据原始列名获取数据（若列名包含空格或括号，请使用 .() 索引）
N = height(dataTable);                 % 10分钟一采样的序列数，通过行数实现
serialNo = (1:N)';                     % 用行号作为序列编号
rawDisplacement = dataTable{:, 2};     % 第2列：位移列
disp(['数据长度：', num2str(N)]);

%% ---- 2. 零值线性插值（必须放在所有滤波之前） ----
zeroIdx = find(rawDisplacement == 0);
disp(['零值点个数：', num2str(length(zeroIdx))]);

interpData = rawDisplacement;
if ~isempty(zeroIdx)
    validIdx = find(rawDisplacement ~= 0);
    interpData(zeroIdx) = interp1(validIdx, rawDisplacement(validIdx), ...
                                  zeroIdx, 'linear', 'extrap');
end

%% ---- 3. 短窗口中值滤波（去除瞬时跳变） ----
winMedian = 9;  % 窗口大小，奇数（跳变尖锐时取 3-7）
medianFiltered = medfilt1(interpData, winMedian); 

%% ---- 4. 小波阈值去噪 ----
% 使用 wdenoise 函数（需要 Wavelet Toolbox）
waveletDenoised = wdenoise(medianFiltered(:)); 
waveletDenoised = waveletDenoised(:);

%% ---- 5. Savitzky‑Golay 滤波（趋势平滑） ----
order = 2;        % 多项式阶数（常用 2-4）
framelen = 27;    % 窗口长度（必须为奇数，典型值 7-21）
sgFiltered = sgolayfilt(waveletDenoised, order, framelen);

%% ---- 6. 绘制对比图 ----
figure('Position', [100, 100, 1400, 900]);

subplot(2,2,1);
plot(serialNo, interpData, 'g-', serialNo, medianFiltered, 'r-', 'LineWidth', 1);
title('① 中值滤波后（去除跳变）'); xlabel('Serial No.'); ylabel('Displacement (mm)');
legend('插值后','中值滤波','Location','best'); grid on;

subplot(2,2,2);
plot(serialNo, medianFiltered, 'r-', serialNo, waveletDenoised, 'b-', 'LineWidth', 1);
title('② 小波阈值去噪后'); xlabel('Serial No.'); ylabel('Displacement (mm)');
legend('中值滤波','小波去噪','Location','best'); grid on;

subplot(2,2,3);
plot(serialNo, waveletDenoised, 'b-', serialNo, sgFiltered, 'k-', 'LineWidth', 1.5);
title('③ S‑G 滤波后（最终结果）'); xlabel('Serial No.'); ylabel('Displacement (mm)');
legend('小波去噪','S‑G滤波','Location','best'); grid on;

subplot(2,2,4);
plot(serialNo, rawDisplacement, 'b-', 'LineWidth', 0.5); hold on;
plot(serialNo, sgFiltered, 'k-', 'LineWidth', 1.5);
title('④ 原始 vs 最终滤波结果'); xlabel('Serial No.'); ylabel('Displacement (mm)');
legend('原始','最终','Location','best'); grid on;

sgtitle('混合滤波流程：插值 → 中值滤波 → 小波去噪 → S‑G 平滑');

%% ---- 7. 保存结果 ----
resultTable = table(serialNo, rawDisplacement, sgFiltered, ...
    'VariableNames', {'SerialNo', 'RawDisplacement', 'FilteredDisplacement'});
writetable(resultTable, 'Filtered_Result.csv');
disp('滤波结果已保存为 Filtered_Result.csv');


%% ---- 8. 输出统计信息（基于插值后数据计算，避免零值干扰） ----
residual = interpData - sgFiltered;   % 使用插值后数据更能反映滤波精度
rmse = sqrt(mean(residual.^2));
snr = 20 * log10(std(interpData) / std(residual));
disp(['RMSE（基于插值后数据）：', num2str(rmse)]);
disp(['信噪比（SNR）：', num2str(snr), ' dB']);
