%% ============================================================
% Q4 速率叠加模型 -- MATLAB仿真测试版
% 读取 ap4.xlsx 的训练集数据，运行速率叠加模型
% ============================================================
% 
% 速率叠加公式:
%   v(t) = v_basic + kr*R_eff(t) + kp*max(0,P-Pcrit) + km*M_cum(t)
%   D(t) = D(0) + cumsum(v)    (积分求位移)
%   爆破脉冲项 beta * (Q / R^3) 单独叠加
% ============================================================

clear; clc; close all;

%% ============================================================
% 用户调参区
% ============================================================

% --- 文件路径 ---
filename = '../ap4.xlsx';
sheet_name = '训练集';

% --- 阶段划分参数 ---
AUTO_PHASE = true;
MANUAL_BREAK1 = 200;
MANUAL_BREAK2 = 800;
SMOOTH_WIN = 50;
RATE_THRESH1 = 0.02;
RATE_THRESH2 = 0.10;
CONTINUOUS_N = 10;

% --- 模型参数（分阶段）---
% 阶段一
v0_1 = 0.002;
kr_1 = 0.0005;
tau_r1 = 30;
Lr_1 = 50;
kp_1 = 0.0001;
Pcrit_1 = 50;
km_1 = 0;
Lm_1 = 50;
beta_1 = 0.1;

% 阶段二
v0_2 = 0.01;
kr_2 = 0.003;
tau_r2 = 150;
Lr_2 = 200;
kp_2 = 0.002;
Pcrit_2 = 40;
km_2 = 0.00005;
Lm_2 = 200;
beta_2 = 0.5;

% 阶段三
v0_3 = 0.05;
kr_3 = 0.008;
tau_r3 = 80;
Lr_3 = 150;
kp_3 = 0.01;
Pcrit_3 = 35;
km_3 = 0.0005;
Lm_3 = 300;
beta_3 = 1.0;

P0 = 30;

%% ============================================================
% 数据读取 -- 使用readtable + table2array直接取数值矩阵
% ============================================================

% 用readtable读取, 保留原始列名
opts = detectImportOptions(filename, 'Sheet', sheet_name);
opts.VariableNamingRule = 'preserve';  % 保留原始列名
T = readtable(filename, opts, 'Sheet', sheet_name);

% 将整个table转为数值矩阵, 自动跳过非数值列(时间)
% 时间列在第1列, 数值列在第2~7列
data_mat = table2array(T(:, 2:end));
n = size(data_mat, 1);

fprintf('数据加载完成: %d 个样本点\n', n);
fprintf('变量: Surface Displacement, Rainfall, Pore Pressure, Microseismic, Blast Dist, Charge\n');

% 提取各列 (与ap4.xlsx列顺序完全对应)
D_real = data_mat(:, 1);  % Surface Displacement (mm)
rain   = data_mat(:, 2);  % Rainfall (mm)
pore   = data_mat(:, 3);  % Pore Water Pressure (kPa)
micro  = data_mat(:, 4);  % Microseismic Event Count
dist   = data_mat(:, 5);  % Blasting Point Distance (m)
charge = data_mat(:, 6);  % Maximum Charge per Segment (kg)

fprintf('位移范围: %.3f ~ %.3f mm\n', min(D_real), max(D_real));

% 缺失值处理 (table2array自动将非数值转为NaN)
nan_idx = isnan(rain);   rain(nan_idx) = 0;
nan_idx = isnan(micro);  micro(nan_idx) = 0;
nan_idx = isnan(dist);   dist(nan_idx) = 0;
nan_idx = isnan(charge); charge(nan_idx) = 0;
% 孔压用前向填充
pore = fillmissing(pore, 'previous');

t_axis = (0:n-1)';

%% ============================================================
% 自动阶段划分（基于速率阈值）
% ============================================================
if AUTO_PHASE
    vel = [0; diff(D_real)];
    vel_smooth = movmean(vel, SMOOTH_WIN);
    b1 = n; b2 = n;
    exceed1 = vel_smooth > RATE_THRESH1;
    for i = 1:(n-CONTINUOUS_N+1)
        if all(exceed1(i:i+CONTINUOUS_N-1))
            b1 = i;
            break;
        end
    end
    exceed2 = vel_smooth > RATE_THRESH2;
    for i = 1:(n-CONTINUOUS_N+1)
        if all(exceed2(i:i+CONTINUOUS_N-1))
            b2 = i;
            break;
        end
    end
    if b1 >= b2
        b1 = max(1, b2 - 200);
    end
else
    b1 = MANUAL_BREAK1;
    b2 = MANUAL_BREAK2;
end

b1 = max(1, min(n, b1));
b2 = max(b1+10, min(n, b2));

fprintf('阶段划分:\n');
fprintf('  阶段一: 1 ~ %d\n', b1);
fprintf('  阶段二: %d ~ %d\n', b1+1, b2);
fprintf('  阶段三: %d ~ %d\n', b2+1, n);

slices = {1:b1, (b1+1):b2, (b2+1):n};

%% ============================================================
% 分阶段速率叠加计算
% ============================================================
v0_list   = [v0_1, v0_2, v0_3];
kr_list   = [kr_1, kr_2, kr_3];
tau_r_list = [tau_r1, tau_r2, tau_r3];
Lr_list   = [Lr_1, Lr_2, Lr_3];
kp_list   = [kp_1, kp_2, kp_3];
Pcrit_list = [Pcrit_1, Pcrit_2, Pcrit_3];
km_list   = [km_1, km_2, km_3];
Lm_list   = [Lm_1, Lm_2, Lm_3];
beta_list = [beta_1, beta_2, beta_3];

D_pred = zeros(n, 1);
current_start_disp = D_real(1);
v_components = zeros(n, 4);  % [v_base, v_rain, v_pore, v_micro]

for ph = 1:3
    idx = slices{ph};
    if isempty(idx), continue; end
    Nph = length(idx);

    rain_ph   = rain(idx);
    pore_ph   = pore(idx);
    micro_ph  = micro(idx);
    dist_ph   = dist(idx);
    charge_ph = charge(idx);

    % 1. 基础蠕变
    v_base = v0_list(ph) * ones(Nph, 1);

    % 2. 降雨有效入渗 (指数衰减记忆)
    Lr  = Lr_list(ph);
    tau = tau_r_list(ph);
    w = exp(-(0:Lr)' / tau);
    w = w / sum(w);
    rain_padded = [zeros(Lr, 1); rain_ph];
    Reff = conv(rain_padded, w, 'same');
    Reff = Reff(Lr+1:end);
    if length(Reff) > Nph
        Reff = Reff(1:Nph);
    elseif length(Reff) < Nph
        Reff = [Reff; zeros(Nph - length(Reff), 1)];
    end
    v_rain = kr_list(ph) * Reff;

    % 3. 孔压项
    exceed = max(0, pore_ph - Pcrit_list(ph));
    v_pore = kp_list(ph) * exceed .* (pore_ph / P0);

    % 4. 微震累积 (滑动窗口求和)
    Lm = Lm_list(ph);
    Mcum = movsum(micro_ph, [Lm, 0], 'Endpoints', 'shrink');
    v_micro = km_list(ph) * Mcum;

    % 5. 爆破脉冲 (直接作为位移增量)
    blast_flag = (dist_ph > 0) & (charge_ph > 0);
    blast_delta = zeros(Nph, 1);
    if any(blast_flag)
        Q = charge_ph(blast_flag);
        R = max(dist_ph(blast_flag), 0.1);
        blast_delta(blast_flag) = beta_list(ph) * (Q ./ (R.^3));
    end

    % 总速率 (四分量叠加)
    v_total = v_base + v_rain + v_pore + v_micro;
    v_total = max(0, v_total);

    % 积分求位移
    D_trend = zeros(Nph, 1);
    D_trend(1) = current_start_disp;
    for i = 2:Nph
        D_trend(i) = D_trend(i-1) + v_total(i-1) + blast_delta(i-1);
    end

    D_pred(idx) = D_trend;
    v_components(idx, :) = [v_base, v_rain, v_pore, v_micro];
    current_start_disp = D_trend(end);
end

%% ============================================================
% 评估与绘图
% ============================================================
res = D_real - D_pred;
rmse = sqrt(mean(res.^2));
mae = mean(abs(res));
r2 = 1 - sum(res.^2) / sum((D_real - mean(D_real)).^2);
nrmse = rmse / (max(D_real) - min(D_real)) * 100;

fprintf('\n========== 评估指标 ==========\n');
fprintf('RMSE  = %.4f mm\n', rmse);
fprintf('MAE   = %.4f mm\n', mae);
fprintf('R^2   = %.4f\n', r2);
fprintf('NRMSE = %.2f %%\n', nrmse);
fprintf('==============================\n');

figure('Position', [50, 50, 1600, 900]);

% (1) 位移拟合
subplot(2,3,1);
hold on;
plot(t_axis, D_real, 'b-', 'LineWidth', 1, 'DisplayName', '实测');
plot(t_axis, D_pred, 'r-', 'LineWidth', 2, 'DisplayName', '模型预测');
xline(b1+0.5, '--g', 'LineWidth', 1.2, 'DisplayName', '阶段边界1');
xline(b2+0.5, '--m', 'LineWidth', 1.2, 'DisplayName', '阶段边界2');
xlabel('时间步 (10min/步)');
ylabel('表面位移 (mm)');
title('速率叠加模型拟合效果');
legend('Location', 'northwest');
grid on; hold off;

% (2) 残差时序
subplot(2,3,2);
plot(t_axis, res, 'k-', 'LineWidth', 0.8);
hold on;
yline(0, '--r', 'LineWidth', 1);
yline(2*std(res), '--', 'Color', [0.5 0.5 0.5]);
yline(-2*std(res), '--', 'Color', [0.5 0.5 0.5]);
xlabel('时间步');
ylabel('残差 (mm)');
title(sprintf('残差时序 (RMSE=%.3fmm)', rmse));
grid on; hold off;

% (3) 残差分布
subplot(2,3,3);
histogram(res, 50, 'Normalization', 'pdf', 'FaceColor', [0.3 0.6 0.9]);
hold on;
x_vals = linspace(min(res), max(res), 200);
y_fit = normpdf(x_vals, mean(res), std(res));
plot(x_vals, y_fit, 'r-', 'LineWidth', 2);
xlabel('残差 (mm)');
ylabel('概率密度');
title(sprintf('残差分布 (MAE=%.3fmm)', mae));
grid on; hold off;

% (4) 速率分量堆叠
subplot(2,3,4);
area(t_axis, max(0, v_components), 'LineWidth', 0.5);
xlabel('时间步');
ylabel('速率 (mm/10min)');
title('速率分量堆叠');
legend({'基础蠕变', '降雨', '孔压', '微震'}, 'Location', 'northwest');
grid on;

% (5) 各阶段平均贡献
subplot(2,3,5);
mean_comp = zeros(3, 4);
for ph = 1:3
    idx = slices{ph};
    if ~isempty(idx)
        mean_comp(ph, :) = mean(v_components(idx, :), 1);
    end
end
bar(mean_comp);
set(gca, 'XTickLabel', {'阶段一', '阶段二', '阶段三'});
ylabel('平均速率 (mm/10min)');
title('各阶段平均速率贡献');
legend({'基础', '降雨', '孔压', '微震'}, 'Location', 'northeast');
grid on;

% (6) 实测vs预测散点图
subplot(2,3,6);
scatter(D_real, D_pred, 5, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
m = min([D_real; D_pred]);
M = max([D_real; D_pred]);
plot([m, M], [m, M], 'r--', 'LineWidth', 1.5);
xlabel('实测位移 (mm)');
ylabel('预测位移 (mm)');
title(sprintf('实测 vs 预测 (R^2=%.4f)', r2));
axis equal; grid on; hold off;

% 保存结果
T_out = table(t_axis, D_real, D_pred, res, ...
    'VariableNames', {'Step', 'Actual_Displacement', ...
    'Predicted_Displacement', 'Residual'});
writetable(T_out, 'simulate_output.csv');
fprintf('结果已保存: simulate_output.csv\n');
