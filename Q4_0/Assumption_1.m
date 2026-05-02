%% ============================================================
% 分段匀加速运动假设检验脚本（速度线性分段，含逐段残差分析）
% 核心：findchangepts 检测速度的线性趋势突变点
% 对每段独立线性拟合 → 速度分段直线 → 积分得位移 → 与实测位移对比
% 输出：第一个分段点作为固定分点给 Assumption_2.m
% ============================================================
clear; clc; close all;

%% ---- 0. 读取数据 ----
dataTable = readtable('Filtered_Result.csv');
t_idx = dataTable.SerialNo;                % 序号
x_filt = dataTable.FilteredDisplacement;   % 滤波后位移 (mm)

N = length(x_filt);
dt = 10;                                   % 采样间隔 (min)
t_min = (t_idx - 1) * dt;                  % 时间轴 (min)

% 读取原始速度（仅 sgolay，未混合滤波）
dataV = readtable('V_A_filtered.csv');
v = dataV.Velocity_mm_min_raw;             % 原始速度 (mm/min)

fprintf('数据长度: %d, 时间跨度: %.1f min\n', N, t_min(end));

%% ---- 1. 核心：findchangepts 检测速度的线性趋势突变点 ----
max_cp = 3;                    % 最多检测 3 个拐点（最多 4 段）
min_dist = round(N * 0.005);   % 最小间隔，防止过于密集

[cp_idx, cp_res] = findchangepts(v, ...
    'MaxNumChanges', max_cp, ...
    'Statistic', 'linear', 'MinDistance',min_dist);  % 灵敏度可调

cp_idx = sort(cp_idx(:)');     % 转为行向量，确保索引顺序
cp_time = t_min(cp_idx);       % 拐点对应的时间 (min)

fprintf('\n检测到的拐点序号: %s\n', mat2str(cp_idx));
fprintf('对应时间(分钟): %s\n', mat2str(cp_time));

%% ---- 2. 分段线性拟合速度（四段独立直线） ----
% 定义分段区间：[1, cp1], [cp1+1, cp2], [cp2+1, cp3], [cp3+1, N]
segments = [1, cp_idx+1; cp_idx, N]';  % 每一行是 [start, end]
if segments(1,1) ~= 1, segments(1,1) = 1; end
if segments(end,2) ~= N, segments(end,2) = N; end

nSeg = size(segments, 1);  % 应为 4

fprintf('\n======== 分段匀加速（速度线性）拟合结果与残差分析 ========\n');
v_fit = zeros(N, 1);       % 拟合速度
coeffs = zeros(nSeg, 2);   % 每段的 [截距 b, 斜率 k] (v = b + k*t)
v_residual = zeros(N, 1);  % 速度拟合残差

for i = 1:nSeg
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end
        continue;
    end
    seg_t = t_min(idx_start:idx_end);
    seg_v = v(idx_start:idx_end);
    
    p = polyfit(seg_t, seg_v, 1);           % p(1) 斜率（加速度），p(2) 截距
    seg_v_fit = polyval(p, seg_t);
    v_fit(idx_start:idx_end) = seg_v_fit;
    v_residual(idx_start:idx_end) = seg_v - seg_v_fit;
    coeffs(i, :) = [p(2), p(1)];           % b, k
    
    % 本段速度残差统计
    seg_res = seg_v - seg_v_fit;
    rmse_v = sqrt(mean(seg_res.^2));
    mae_v  = mean(abs(seg_res));
    max_res_v = max(abs(seg_res));
    
    fprintf('\n阶段 %d（t = %.1f~%.1f min，点 %d~%d）:\n', ...
            i, seg_t(1), seg_t(end), idx_start, idx_end);
    fprintf('  速度公式: v(t) = %.6f + %.6f * t  (加速度 = %.6f mm/min²)\n', ...
            p(2), p(1), p(1));
    fprintf('  速度残差分析: RMSE = %.6f mm/min, MAE = %.6f mm/min, 最大绝对残差 = %.6f mm/min\n', ...
            rmse_v, mae_v, max_res_v);
end

%% ---- 3. 由拟合速度积分得到拟合位移 ----
x_model = zeros(N, 1);
x_model(1) = x_filt(1);  % 初始位移与滤波数据对齐
for i = 2:N
    % 梯形积分
    x_model(i) = x_model(i-1) + 0.5*(v_fit(i-1) + v_fit(i)) * dt;
end

% 位移残差
residual_disp = x_filt - x_model;

%% ---- 4. 绘图对比 ----
figure('Position', [100, 100, 1400, 900]);

subplot(3,1,1);
plot(t_min, v, 'k-', 'LineWidth', 1.2); hold on;
plot(t_min, v_fit, 'r-', 'LineWidth', 1.5);
for i = 1:length(cp_idx)
    xline(cp_time(i), '--', 'LineWidth', 1.0);
end
ylabel('速度 (mm/min)');
title('速度：原始速度 vs 分段线性拟合');
legend('原始速度', '拟合速度（按段线性）', '拐点', 'Location', 'best');
grid on;

subplot(3,1,2);
plot(t_min, x_filt, 'b-', 'LineWidth', 1.2); hold on;
plot(t_min, x_model, 'r--', 'LineWidth', 1.5);
for i = 1:length(cp_idx)
    xline(cp_time(i), '--', 'LineWidth', 1.0);
end
ylabel('位移 (mm)');
title('位移：实测位移 vs 拟合位移（分段积分）');
legend('实测位移', '拟合位移（按段积分）', '拐点', 'Location', 'best');
grid on;

subplot(3,1,3);
plot(t_min, residual_disp, 'k-', 'LineWidth', 0.8);
xlabel('时间 t (min)');
ylabel('位移残差 (mm)');
title('位移拟合残差（实测 - 模型）');
grid on;

sgtitle('分段匀加速模型尝试性探究(纯linear)');

%% ---- 5. 逐段位移残差分析 ----
fprintf('\n======== 各阶段位移拟合残差分析 ========\n');
for i = 1:nSeg
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end
        continue;
    end
    seg_res_disp = residual_disp(idx_start:idx_end);
    rmse_d = sqrt(mean(seg_res_disp.^2));
    mae_d  = mean(abs(seg_res_disp));
    max_d  = max(abs(seg_res_disp));
    
    fprintf('阶段 %d（t = %.1f ~ %.1f min）: 位移 RMSE = %.6f mm, MAE = %.6f mm, 最大绝对残差 = %.6f mm\n', ...
            i, t_min(idx_start), t_min(idx_end), rmse_d, mae_d, max_d);
end

%% ---- 6. 整体拟合优度评价 ----
SS_res = sum((x_filt - x_model).^2);
SS_tot = sum((x_filt - mean(x_filt)).^2);
R_sq   = 1 - SS_res/SS_tot;
RMSE   = sqrt(mean((x_filt - x_model).^2));
MAE    = mean(abs(x_filt - x_model));

fprintf('\n======== 整体位移拟合评价 ========\n');
fprintf('决定系数 R²  : %.6f\n', R_sq);
fprintf('均方根误差 RMSE : %.4f mm\n', RMSE);
fprintf('平均绝对误差 MAE : %.4f mm\n', MAE);
fprintf('最大绝对残差     : %.4f mm\n', max(abs(x_filt - x_model)));

% 定性结论
if R_sq > 0.999 && max(abs(x_filt - x_model)) < 0.5  % 阈值可调整
    fprintf('\n>>> 评估：分段匀加速模型对位移的拟合精度极高，\n');
    fprintf('    该系统可近似为分阶段匀加速运动。\n');
else
    fprintf('\n>>> 评估：位移残差明显，分段匀加速模型不能完美描述实际运动，\n');
    fprintf('    系统可能存在加速度连续变化的过程（非恒定加速度）。\n');
end

%% ---- 7. 前三段整体位移拟合评价 ----
first3_mask = false(N, 1);
for i = 1:min(3, nSeg)
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start <= idx_end
        first3_mask(idx_start:idx_end) = true;
    end
end
residual_first3 = residual_disp(first3_mask);
x_filt_first3   = x_filt(first3_mask);

SS_res3 = sum(residual_first3.^2);
SS_tot3 = sum((x_filt_first3 - mean(x_filt_first3)).^2);
R_sq3   = 1 - SS_res3/SS_tot3;
RMSE3   = sqrt(mean(residual_first3.^2));
MAE3    = mean(abs(residual_first3));

fprintf('\n======== 前三段整体位移拟合评价 ========\n');
fprintf('决定系数 R²  : %.6f\n', R_sq3);
fprintf('均方根误差 RMSE : %.4f mm\n', RMSE3);
fprintf('平均绝对误差 MAE : %.4f mm\n', MAE3);
fprintf('最大绝对残差     : %.4f mm\n', max(abs(residual_first3)));

%% ---- 8. 输出关键参数给 Assumption_2.m ---- 
% 第一个分段点索引和对应时间，供 Assumption_2.m 作为固定分点使用
first_cp_idx = cp_idx(1);
first_cp_time_min = t_min(first_cp_idx);
paramTable = table(first_cp_idx, first_cp_time_min, ...
    'VariableNames', {'FirstSplitIndex', 'FirstSplitTime_min'});
writetable(paramTable, 'Assumption1_Params.xlsx');
fprintf('\n======== 已输出参数到 Assumption1_Params.xlsx ========\n');
fprintf('固定分点索引: %d, 时间: %.1f min\n', first_cp_idx, first_cp_time_min);
