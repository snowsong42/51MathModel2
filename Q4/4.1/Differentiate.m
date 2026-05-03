%% ============================================================
% 位移序列速度与加速度分析
% 输入：Filtered_Result.csv（滤波后的位移）
% 功能：① sgolay计算速度；② 混合滤波
% 输出：V_A_filtered.csv
% ============================================================
clear; clc; close all;

%% ---- 0. 加载滤波后位移数据 ----
dataTable = readtable('Filtered_Result.csv');

serialNo = dataTable.SerialNo;
x = dataTable.FilteredDisplacement;      % 滤波后位移 (mm)
dt = 10;                                 % 采样间隔(min)
t = ((1:length(x))' - 1) * dt;           % 时间轴 (min)，从0开始

fprintf('数据点数: %d \n', length(x));
fprintf('时间间隔: %.1f min\n', dt);

%% ---- 1. sgolay 参数设定 ----
order = 3;       % 多项式阶数
framelen = 27;   % 窗口长度（奇数）

[~, g] = sgolay(order, framelen);
m = (framelen - 1) / 2;   % 半边窗口长度

%% ---- 2. sgolay 计算速度 ----
v_raw = conv(x, factorial(1) / (-dt)^1 * g(:, 2), 'same');
% 修正边缘效应
v_raw(1:m) = v_raw(m+1);  v_raw(end-m+1:end) = v_raw(end-m);

%% ---- 2.5 对速度进行混合滤波 ----
winMedian = 100;        % 中值滤波窗口
sg_order  = 2;          % S‑G 多项式阶数
sg_framelen = 45;       % S‑G 窗口长度（奇数）

fprintf('速度滤波参数: winMedian=%d, sg_order=%d, sg_framelen=%d\n', winMedian, sg_order, sg_framelen);

v = v_raw;
v = medfilt1(v, winMedian);
v = wdenoise(v(:)); v = v(:);
v = sgolayfilt(v, sg_order, sg_framelen);

fprintf('加速度混合滤波完成。\n');

%% ---- 3. 绘制速度 ----
figure('Position', [100, 100, 1200, 500]);
plot(t, v, 'b-', 'LineWidth', 1.2);
xlabel('时间 t (min)');
ylabel('速度 (mm/min)');
title('速度曲线');
grid on;

%% ---- 4. 保存结果 ----
resultTable = table(serialNo, x, v, ...
    'VariableNames', {'SerialNo', 'Displacement_mm', ...
                      'Velocity_mm_min'});
writetable(resultTable, 'V_A_filtered.csv');
fprintf('速度已保存为V_A_filtered.csv。\n');
