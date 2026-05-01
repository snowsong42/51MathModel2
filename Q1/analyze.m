%% 清理环境
clear; clc; close all;

%% 1. 读取数据
filename = 'D:\project\pythonProject\MathModel\51MathModel2\Q1\Attachment 1.xlsx';
data = readtable(filename, 'VariableNamingRule', 'preserve');

% 时间列（注意根据实际列名调整）
time = datetime(data.Time, 'InputFormat', 'yyyy-MM-dd HH:mm');
A = data.('Data A (Optical Fiber Displacement Sensor Data, mm)');
B = data.('Data B (Vibrating Wire Displacement Sensor Data, mm)');

% 转换为相对时间（小时），方便趋势分析
t_hours = hours(time - time(1));
n = length(A);

%% 2. 创建综合诊断图形窗口
figure('Position', [50, 50, 1400, 900]);

% ----- 子图1：原始时序对比 -----
subplot(2,3,1);
plot(time, A, 'b-', 'LineWidth', 1.2); hold on;
plot(time, B, 'r-', 'LineWidth', 1.2);
xlabel('时间'); ylabel('位移 (mm)');
title('原始时序对比 (A:蓝, B:红)');
legend('Data A (光纤)', 'Data B (振弦)', 'Location','best');
grid on;

% ----- 子图2：差值序列 Δ = A - B -----
Delta = A - B;
subplot(2,3,2);
plot(time, Delta, 'k.-', 'MarkerSize', 6);
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('时间'); ylabel('\Delta = A - B (mm)');
title('差值序列');
grid on;

% ----- 子图3：A vs B 散点图及线性拟合 -----
subplot(2,3,3);
scatter(B, A, 10, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
% 参考线 y = x
limits = [min([A;B]) max([A;B])];
plot(limits, limits, 'r--', 'LineWidth', 1.5, 'DisplayName', 'y = x');
% 线性拟合
coeff = polyfit(B, A, 1);
B_line = linspace(limits(1), limits(2), 100);
A_line = polyval(coeff, B_line);
plot(B_line, A_line, 'g-', 'LineWidth', 2, ...
    'DisplayName', sprintf('拟合: A=%.3f·B + %.3f', coeff(1), coeff(2)));
xlabel('Data B (mm)'); ylabel('Data A (mm)');
title('A 与 B 散点图');
legend('Location','best'); grid on;
axis equal;

% ----- 子图4：差值趋势分析 -----
subplot(2,3,4);
plot(t_hours, Delta, '.', 'MarkerSize', 6);
hold on;
p_delta = polyfit(t_hours, Delta, 1);
Delta_trend = polyval(p_delta, t_hours);
plot(t_hours, Delta_trend, 'r-', 'LineWidth', 2, ...
    'DisplayName', sprintf('趋势: Δ = %.3f + %.3f·t', p_delta(2), p_delta(1)));
xlabel('时间 (小时)'); ylabel('\Delta (mm)');
title('差值随时间的变化趋势');
legend('Location','best'); grid on;

% ----- 子图5：互相关分析 -----
subplot(2,3,5);
[corr_vals, lags] = xcorr(A - mean(A), B - mean(B), 'coeff');
plot(lags, corr_vals, 'b-', 'LineWidth', 1.2);
xlabel('滞后 (样本点, 每点代表10分钟)');
ylabel('互相关系数');
title('互相关函数');
grid on;
% 标注最大相关滞后
[~, idx_max] = max(corr_vals);
best_lag = lags(idx_max);
hold on;
plot(best_lag, corr_vals(idx_max), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('互相关', sprintf('最大相关滞后 = %d 样本点', best_lag), 'Location','best');

% ----- 子图6：频谱对比 -----
subplot(2,3,6);
Fs = 1 / (10*60); % 采样频率 Hz (10分钟间隔)
f_axis = (0:floor(n/2)-1) * (Fs / n);
A_fft = abs(fft(A - mean(A))) / n;
B_fft = abs(fft(B - mean(B))) / n;
plot(f_axis, A_fft(1:floor(n/2)), 'b-', 'LineWidth', 1);
hold on;
plot(f_axis, B_fft(1:floor(n/2)), 'r-', 'LineWidth', 1);
xlabel('频率 (Hz)'); ylabel('幅值');
title('频谱对比 (A:蓝, B:红)');
legend('Data A', 'Data B'); grid on;
xlim([0 max(f_axis)]);

sgtitle('位移传感器数据 A 与 B 偏移类型诊断', 'FontSize', 14);

%% 3. 输出关键统计量
fprintf('================= 偏移类型诊断统计 =================\n');
fprintf('差值 Δ 的均值     ：%8.4f mm\n', mean(Delta));
fprintf('差值 Δ 的标准差   ：%8.4f mm\n', std(Delta));
fprintf('最佳互相关滞后     ：%d 样本点 (%.1f 分钟)\n', best_lag, best_lag*10);
fprintf('A vs B 线性拟合    ：A = %.4f·B + %.4f\n', coeff(1), coeff(2));
fprintf('差值趋势斜率       ：%.4f mm/小时\n', p_delta(1));
fprintf('====================================================\n');