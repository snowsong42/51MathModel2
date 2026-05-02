%% ============================================================
% 位移序列速度与加速度分析
% 输入：Filtered_Result.csv（滤波后的位移）
% 功能：① 可选择起止时间分析；② 加速度独立混合滤波（速度仅 sgolay）
% 输出：上下两幅图（速度、加速度）
% ============================================================
clear; clc; close all;

%% ---- 0. 加载滤波后位移数据 ----
dataTable = readtable('Filtered_Result.csv');
serialNo = dataTable.SerialNo;
x = dataTable.FilteredDisplacement;      % 滤波后位移 (mm)

N = length(x);
dt = 10;   % 采样间隔（若实际为 10 min，改为 dt = 10；单位自行定义）

%% ---- 0.1 选择分析的时间段（按序号） ----
% 修改下面两行参数，即可截取需要分析的数据段
start_idx = 1;        % 起始序号
end_idx   = N;        % 结束序号
if start_idx < 1, start_idx = 1; end
if end_idx > N, end_idx = N; end
% 截取数据
idx_range = start_idx:end_idx;
t = (idx_range' - 1) * dt;              % 时间轴 (min)，从0开始
x = x(idx_range);
serialNo = serialNo(idx_range);

N_used = length(x);
fprintf('数据点数: %d (总长 %d), 使用范围: %d ~ %d\n', N_used, N, start_idx, end_idx);
fprintf('时间间隔: %.1f min\n', dt);

%% ---- 1. sgolay 参数设定 ----
order = 3;       % 多项式阶数
framelen = 27;   % 窗口长度（奇数）

[~, g] = sgolay(order, framelen);
m = (framelen - 1) / 2;   % 半边窗口长度

%% ---- 2. 计算原始速度与加速度 ----
v = conv(x, factorial(1) / (-dt)^1 * g(:, 2), 'same');
a_raw = conv(x, factorial(2) / (-dt)^2 * g(:, 3), 'same');

% 修正边缘效应
v(1:m) = v(m+1);          v(end-m+1:end) = v(end-m);
a_raw(1:m) = a_raw(m+1);  a_raw(end-m+1:end) = a_raw(end-m);

%% ---- 2.5 对加速度进行混合滤波（速度不滤波） ----
winMedian = 100;        % 中值滤波窗口
sg_order  = 2;          % S‑G 多项式阶数
sg_framelen = 45;       % S‑G 窗口长度（奇数）

fprintf('加速度滤波参数: winMedian=%d, sg_order=%d, sg_framelen=%d\n', winMedian, sg_order, sg_framelen);

a = a_raw;
a = medfilt1(a, winMedian);
a = wdenoise(a(:)); a = a(:);
a = sgolayfilt(a, sg_order, sg_framelen);

fprintf('加速度混合滤波完成。\n');

%% ---- 3. 绘制速度与加速度（上下两幅图） ----
figure('Position', [100, 100, 1200, 500]);

subplot(2,1,1);
plot(t, v, 'r-', 'LineWidth', 1.2);
xlabel(['时间 t (min)  [范围：', num2str(start_idx), ' ~ ', num2str(end_idx), ']']);
ylabel('速度 (mm/min)');
title('速度曲线（sgolay 后，未混合滤波）');
grid on;

subplot(2,1,2);
plot(t, a, 'b-', 'LineWidth', 1.2); hold on;
plot(t, zeros(size(t)), 'k--', 'LineWidth', 0.8);
xlabel('时间 t (min)');
ylabel('加速度 (mm/min²)');
title('加速度曲线（sgolay + 混合滤波后）');
legend('加速度', '零线', 'Location', 'best');
grid on;

sgtitle('位移序列的速度与加速度分析');

%% ---- 4. 保存结果 ----
resultTable = table(serialNo, x, v, a, ...
    'VariableNames', {'SerialNo', 'Displacement_mm', ...
                      'Velocity_mm_min_raw', ...
                      'Acceleration_mm_min2_raw'});
writetable(resultTable, 'V_A_filtered.csv');
saveas(gcf, 'V_A_filtered.png');
fprintf('速度（原始）与加速度（混合滤波后）已保存为 CSV 和 PNG。\n');
