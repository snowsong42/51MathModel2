clear; clc; close all;

%% ---- 读取数据 ----
dataTable = readtable('Filtered_Result.csv');
t_idx = dataTable.SerialNo;
x_fact = dataTable.FilteredDisplacement;
v_data = readtable('V_A_filtered.csv');
v = v_data.Velocity_mm_min;

N = length(x_fact);
dt = 10;                             % 采样间隔 min
t_min = t_idx * dt;           % 时间轴 (min)

%% ---- 固定3段，用 RMS 找两个拐点 ----
cp_idx = findchangepts(v, 'MaxNumChanges', 2, 'Statistic', 'rms');
cp_time = t_min(cp_idx);

% 直接构造3段区间索引
seg1 = 1:cp_idx(1);
seg2 = (cp_idx(1)+1):cp_idx(2);
seg3 = (cp_idx(2)+1):N;

%% ---- 分段拟合：前两段多项式，第三段指数 ----
ord = [1, 2];          % 前两段多项式阶数
v_fit = zeros(N,1);
x_model = zeros(N,1);

% ----- 第1段：1次多项式 -----
idx = seg1; ti = t_min(idx); xi = x_fact(idx); t0 = ti(1); x0 = xi(1);
A = zeros(length(ti), ord(1)+1);
for k = 0:ord(1)
    A(:, ord(1)+1-k) = (ti.^(k+1) - t0^(k+1)) / (k+1);
end
p1 = A \ (xi - x0);
v_fit(idx) = polyval(p1, ti);
x_model(idx) = x0 + A * p1;

% ----- 第2段：2次多项式 -----
idx = seg2; ti = t_min(idx); xi = x_fact(idx); t0 = ti(1); x0 = xi(1);
A = zeros(length(ti), ord(2)+1);
for k = 0:ord(2)
    A(:, ord(2)+1-k) = (ti.^(k+1) - t0^(k+1)) / (k+1);
end
p2 = A \ (xi - x0);
v_fit(idx) = polyval(p2, ti);
x_model(idx) = x0 + A * p2;

% ----- 第3段：双指数模型（位移）-----
idx = seg3; ti = t_min(idx); xi = x_fact(idx);
[exp_fit, gof_exp] =  fit(ti(:), xi(:), fittype('exp2'), 'Algorithm', 'Levenberg-Marquardt');
coeff = coeffvalues(exp_fit);   % [a, b, c, d]
a = coeff(1); b = coeff(2); c = coeff(3); d = coeff(4);
v_fit(idx) = a*b*exp(b*ti) + c*d*exp(d*ti);   % 速度 = 位移对时间的导数
x_model(idx) = feval(exp_fit, ti);

% 位移残差
residual_disp = x_fact - x_model;

%% ---- 输出各区段信息 ----
fprintf('数据长度: %d, 时间跨度: %.1f min\n', N, t_min(end));
fprintf('拐点索引: %s\n', mat2str(cp_idx));
fprintf('拐点时间: %s min\n', mat2str(cp_time));
fprintf('前两段多项式阶数: %s，第三段为双指数模型\n', mat2str(ord));

segnames = {'第1段','第2段','第3段'};
seg_cell = {seg1, seg2, seg3};

% --- 第1段输出 ---
fprintf('\n阶段 1(t = %.1f~%.1f min , %.2f~%.2f 天，阶数 %d):\n', ...
    t_min(seg1(1)), t_min(seg1(end)), t_min(seg1(1))/1440, t_min(seg1(end))/1440, ord(1));
fprintf('  v(t) = %.6f + %.6e * t\n', p1(2), p1(1));
fprintf('  等效加速度 = %.6e mm/min²\n', p1(1));
res_d1 = x_fact(seg1) - x_model(seg1);
fprintf('  位移(主目标): RMSE=%.6f mm, MAE=%.6f mm, Max=%.6f mm\n', ...
    sqrt(mean(res_d1.^2)), mean(abs(res_d1)), max(abs(res_d1)));
if ~isempty(v)
    seg_v1 = v(seg1); res_v1 = seg_v1 - v_fit(seg1);
    fprintf('  速度残差（辅助）: RMSE=%.6f, MAE=%.6f, Max=%.6f mm/min\n', ...
        sqrt(mean(res_v1.^2)), mean(abs(res_v1)), max(abs(res_v1)));
end

% --- 第2段输出 ---
fprintf('\n阶段 2(t = %.1f~%.1f min , %.2f~%.2f 天，阶数 %d):\n', ...
    t_min(seg2(1)), t_min(seg2(end)), t_min(seg2(1))/1440, t_min(seg2(end))/1440, ord(2));
fprintf('  v(t) = %.6e * t^2 + %.6e * t + %.6f\n', p2(1), p2(2), p2(3));
fprintf('  a(t) = %.6e * t + %.6e\n', 2*p2(1), p2(2));
res_d2 = x_fact(seg2) - x_model(seg2);
fprintf('  位移(主目标): RMSE=%.6f mm, MAE=%.6f mm, Max=%.6f mm\n', ...
    sqrt(mean(res_d2.^2)), mean(abs(res_d2)), max(abs(res_d2)));
if ~isempty(v)
    seg_v2 = v(seg2); res_v2 = seg_v2 - v_fit(seg2);
    fprintf('  速度残差（辅助）: RMSE=%.6f, MAE=%.6f, Max=%.6f mm/min\n', ...
        sqrt(mean(res_v2.^2)), mean(abs(res_v2)), max(abs(res_v2)));
end

% --- 第3段输出（指数模型）---
fprintf('\n阶段 3(t = %.1f~%.1f min , %.2f~%.2f 天，双指数模型):\n', ...
    t_min(seg3(1)), t_min(seg3(end)), t_min(seg3(1))/1440, t_min(seg3(end))/1440);
fprintf('  x(t) = %.6e*exp(%.6f*t) + %.6e*exp(%.6f*t)\n', a, b, c, d);
fprintf('  v(t) = %.6e*exp(%.6f*t) + %.6e*exp(%.6f*t)\n', a*b, b, c*d, d);
fprintf('  拟合优度: R² = %.4f, RMSE = %.4f mm\n', gof_exp.rsquare, gof_exp.rmse);
res_d3 = x_fact(seg3) - x_model(seg3);
fprintf('  位移残差: RMSE=%.6f mm, MAE=%.6f mm, Max=%.6f mm\n', ...
    sqrt(mean(res_d3.^2)), mean(abs(res_d3)), max(abs(res_d3)));
if ~isempty(v)
    seg_v3 = v(seg3); res_v3 = seg_v3 - v_fit(seg3);
    fprintf('  速度残差（辅助）: RMSE=%.6f, MAE=%.6f, Max=%.6f mm/min\n', ...
        sqrt(mean(res_v3.^2)), mean(abs(res_v3)), max(abs(res_v3)));
end

%% ---- 绘图 ----
% 颜色定义
seg_bg = {[0.7 1 0.7], [1 1 0.2], [1 0.7 0.7]};
seg_line = {[0 0.6 0], [1 0.8 0.2], [0.8 0 0]};

% 图1：位移 + 区间着色
figure('Position', [100, 100, 1000, 500]);
hold on;
plot(t_min, x_fact, 'k-', 'LineWidth', 1.2);
yl = ylim;
tg1 = [t_min(seg1(1)) t_min(seg1(end))];
tg2 = [t_min(seg2(1)) t_min(seg2(end))];
tg3 = [t_min(seg3(1)) t_min(seg3(end))];
patch([tg1(1) tg1(2) tg1(2) tg1(1)], [yl(1) yl(1) yl(2) yl(2)], seg_bg{1}, 'EdgeColor','none','FaceAlpha',0.3);
patch([tg2(1) tg2(2) tg2(2) tg2(1)], [yl(1) yl(1) yl(2) yl(2)], seg_bg{2}, 'EdgeColor','none','FaceAlpha',0.3);
patch([tg3(1) tg3(2) tg3(2) tg3(1)], [yl(1) yl(1) yl(2) yl(2)], seg_bg{3}, 'EdgeColor','none','FaceAlpha',0.3);
xline(t_min(cp_idx(1)), '--k', 'LineWidth', 1.0);
xline(t_min(cp_idx(2)), '--k', 'LineWidth', 1.0);
for i = 1:2
    t_cp = t_min(cp_idx(i));
    d_cp = t_cp/1440;
    text(t_cp, yl(2)-0.05*(yl(2)-yl(1)), sprintf('%.2f天', d_cp), ...
        'HorizontalAlignment','left','VerticalAlignment','top', ...
        'FontSize',9,'BackgroundColor','w','EdgeColor','k');
end
xlabel('时间 t (min)'); ylabel('位移 (mm)');
title('分段结果');
legend('位移', segnames{:}, 'Location','best');
grid on; hold off;

% 图2：三行子图（速度、位移、残差）
figure('Position', [100, 100, 1400, 900]);

subplot(3,1,1); hold on;
plot(t_min, v, 'Color',[0.2 0.2 0.2], 'LineWidth',2);
plot(t_min(seg1), v_fit(seg1), 'Color', seg_line{1}, 'LineWidth',1.5);
plot(t_min(seg2), v_fit(seg2), 'Color', seg_line{2}, 'LineWidth',1.5);
plot(t_min(seg3), v_fit(seg3), 'Color', seg_line{3}, 'LineWidth',1.5);
xline(t_min(cp_idx(1)), '--'); xline(t_min(cp_idx(2)), '--');
ylabel('速度 (mm/min)'); title('速度：测算速度-模型拟合速度');
legend('测算速度','拟合速度（按段着色）'); grid on;

subplot(3,1,2); hold on;
plot(t_min, x_fact, 'Color',[0.2 0.2 0.2], 'LineWidth',2);
plot(t_min(seg1), x_model(seg1), 'Color', seg_line{1}, 'LineWidth',1.5);
plot(t_min(seg2), x_model(seg2), 'Color', seg_line{2}, 'LineWidth',1.5);
plot(t_min(seg3), x_model(seg3), 'Color', seg_line{3}, 'LineWidth',1.5);
xline(t_min(cp_idx(1)), '--'); xline(t_min(cp_idx(2)), '--');
ylabel('位移 (mm)'); title('位移：测算位移-模型拟合位移');
legend('测算位移','拟合位移（按段着色）'); grid on;

subplot(3,1,3); hold on;
plot(t_min(seg1), residual_disp(seg1), 'Color', seg_line{1}, 'LineWidth',0.8);
plot(t_min(seg2), residual_disp(seg2), 'Color', seg_line{2}, 'LineWidth',0.8);
plot(t_min(seg3), residual_disp(seg3), 'Color', seg_line{3}, 'LineWidth',0.8);
xline(t_min(cp_idx(1)), '--'); xline(t_min(cp_idx(2)), '--');
xlabel('时间 t (min)'); ylabel('位移残差 (mm)');
title('位移拟合残差'); grid on;

sgtitle('模型验证');

%% ---- 整体统计 ----
SS_res = sum(residual_disp.^2);
SS_tot = sum((x_fact - mean(x_fact)).^2);
R_sq = 1 - SS_res/SS_tot;
RMSE = sqrt(mean(residual_disp.^2));
MAE  = mean(abs(residual_disp));
fprintf('\n======== 整体位移拟合评价 ========\n');
fprintf('决定系数 R²  : %.9f\n', R_sq);
fprintf('均方根误差 RMSE : %.4f mm\n', RMSE);
fprintf('平均绝对误差 MAE : %.4f mm\n', MAE);
fprintf('最大绝对残差     : %.4f mm\n', max(abs(residual_disp)));

%% ============================================================
% [新增] 统一输出 ap4_stage.xlsx
% 功能：① 给训练集标注阶段 ② Time转序列号
%       ③ 添加速度 ④ 统一列名 ⑤ 输出双sheet
% ============================================================

%% ---- 1. 读取原始训练集数据 ----
trainRaw = readtable('Attachment 4.xlsx', 'Sheet', '训练集');
N_train = height(trainRaw);

% 标注阶段 (利用已有的 cp_idx 拐点)
stage_train = zeros(N_train, 1);
stage_train(seg1) = 1;
stage_train(seg2) = 2;
stage_train(seg3) = 3;

% Time 转序列号
time_serial_train = (1:N_train)';

% 提取各列 (按照原列名顺序)
% 原列名: Time, Surface Displacement (mm), Rainfall (mm), Pore Water Pressure (kPa),
%         Microseismic Event Count, Blasting Point Distance (m), Maximum Charge per Segment (kg)
% 映射: a=Rainfall, b=Pore Water Pressure, c=Microseismic Event Count,
%       d=Blasting Point Distance, e=Maximum Charge per Segment, SD=Surface Displacement
col_rainfall_idx   = 3;  % Rainfall (mm)
col_pwp_idx        = 4;  % Pore Water Pressure (kPa)
col_mec_idx        = 5;  % Microseismic Event Count
col_bpd_idx        = 6;  % Blasting Point Distance (m)
col_mcs_idx        = 7;  % Maximum Charge per Segment (kg)
col_sd_idx         = 2;  % Surface Displacement (mm)

a_train = trainRaw{:, col_rainfall_idx};
b_train = trainRaw{:, col_pwp_idx};
c_train = trainRaw{:, col_mec_idx};
d_train = trainRaw{:, col_bpd_idx};
e_train = trainRaw{:, col_mcs_idx};
SD_train = x_fact;  % 使用滤波后的位移作为训练集的 SD 列

% 从 V_A_filtered.csv 读取速度
v_data = readtable('V_A_filtered.csv');
V_train = v_data.Velocity_mm_min;

% 构建训练集表格 (统一列名)
train_out = table(time_serial_train, stage_train, a_train, b_train, c_train, d_train, e_train, SD_train, V_train, ...
    'VariableNames', {'Time', 'Stage', 'a', 'b', 'c', 'd', 'e', 'SD', 'V'});

%% ---- 2. 读取实验集数据 ----
expRaw = readtable('Attachment 4.xlsx', 'Sheet', '实验集');
N_exp = height(expRaw);

% 实验集已有 Stage Label 列，列顺序为: Time, Stage Label, Surface Displacement (mm),
% Rainfall (mm), Pore Water Pressure (kPa), Microseismic Event Count,
% Blasting Point Distance (m), Maximum Charge per Segment (kg)
% Time 转序列号
time_serial_exp = (1:N_exp)';

% 提取实验集 Stage
stage_exp = expRaw{:, 2};  % Stage Label 列 (第2列)

% 提取 a,b,c,d,e
a_exp = expRaw{:, 4};  % Rainfall (mm) 第4列
b_exp = expRaw{:, 5};  % Pore Water Pressure (kPa) 第5列
c_exp = expRaw{:, 6};  % Microseismic Event Count 第6列
d_exp = expRaw{:, 7};  % Blasting Point Distance (m) 第7列
e_exp = expRaw{:, 8};  % Maximum Charge per Segment (kg) 第8列

% 实验集 SD 和 V 留空 (NaN)
SD_exp = NaN(N_exp, 1);
V_exp = NaN(N_exp, 1);

% 构建实验集表格
exp_out = table(time_serial_exp, stage_exp, a_exp, b_exp, c_exp, d_exp, e_exp, SD_exp, V_exp, ...
    'VariableNames', {'Time', 'Stage', 'a', 'b', 'c', 'd', 'e', 'SD', 'V'});

%% ---- 3. 输出到 ap4_stage.xlsx ----
outputFile = 'ap4_stage.xlsx';
writetable(train_out, outputFile, 'Sheet', '训练集');
writetable(exp_out, outputFile, 'Sheet', '实验集');
fprintf('\n===== 已输出 %s =====\n', outputFile);
fprintf('训练集 %d 行, 实验集 %d 行\n', N_train, N_exp);
fprintf('列名: Time  Stage  a  b  c  d  e  SD  V\n');
fprintf('其中 a=Rainfall, b=Pore Water Pressure, c=Microseismic Event Count,\n');
fprintf('     d=Blasting Point Distance, e=Maximum Charge per Segment\n');
fprintf('     SD=Surface Displacement, V=Velocity\n');
