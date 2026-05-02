%% ============================================================
% 混合分段检测 + 固定分点（79570 min）自适应多项式拟合模型
%  前半段 (0~79570 min) 用 'rms' 检测 1 拐点
%  79570 本身作为固定拐点
%  后半段 (>79570 min) 用 'linear' 检测 2 拐点
%  总共 4 拐点 → 5 段，各段多项式阶数可独立设定
% ============================================================
clear; clc; close all;

%% ---- 0. 读取数据 ----
dataTable = readtable('Filtered_Result.csv');
t_idx = dataTable.SerialNo;
x_fact = dataTable.FilteredDisplacement;
dataV = readtable('V_A_filtered.csv');
v = dataV.Velocity_mm_min_filtered;

N = length(x_fact);
dt = 10;
t_min = (t_idx - 1) * dt;

%% ---- 1. 混合分段点检测 (0~split_t_fixed用rms, >split_t_fixed用linear) ----
% 固定分段点时间（min）
split_t_fixed = 87350;
% 找到对应索引（取第一个不小于该时间的点）
split_idx = find(t_min >= split_t_fixed, 1, 'first');
if isempty(split_idx)
    error('未找到 t = %.1f min 对应的数据点，请检查时间轴。', split_t_fixed);
end

% 前半段：用 RMS 统计量检测 1 个拐点
cp1_local = findchangepts(x_fact(1:split_idx), ...
    'MaxNumChanges', 1, ...
    'Statistic', 'std');

% 后半段：用 linear 统计量检测 2 个拐点
v_part2 = v(split_idx+1:end);
min_dist2 = max(1, round((N - split_idx) * 0.01));
cp2_local = findchangepts(v_part2, ...
    'MaxNumChanges', 2, ...
    'Statistic', 'linear', ...
    'MinDistance', min_dist2);

% 将后半段的局部索引映射回全局索引
if isempty(cp2_local)
    cp2_global = [];
else
    cp2_global = cp2_local + split_idx;
end

% 合并拐点：前半段检测点 + 固定分点 + 后半段检测点
cp_idx = [cp1_local(:)', split_idx, cp2_global(:)'];
cp_idx = unique(cp_idx);                % 去重（固定点可能被检测到）
cp_idx = sort(cp_idx);
cp_idx = cp_idx(cp_idx > 1 & cp_idx < N);  % 去掉边界无效点
cp_time = t_min(cp_idx);

% 确保 cp_idx 为行向量，防止后续拼接维度错误
cp_idx = cp_idx(:)';
nCp = length(cp_idx);

% 构造分段区间（段数 = 拐点数 + 1），每行为 [起点, 终点]
% 使用循环逐行构造，避免 vertcat 维度不匹配问题
nSeg = nCp + 1;
segments = zeros(nSeg, 2);
if nCp >= 1
    segments(1, :) = [1, cp_idx(1)];
    for i = 2:nCp
        segments(i, :) = [cp_idx(i-1) + 1, cp_idx(i)];
    end
    segments(nSeg, :) = [cp_idx(end) + 1, N];
else
    % 无拐点：整段从 1 到 N
    segments = [1, N];
    nSeg = 1;
end


% ---- 各段多项式阶数（可按需修改，长度必须等于 nSeg）----
poly_order = [1, 2,2, 2, 3];   % 请根据实际需要修改

fprintf('数据长度: %d, 时间跨度: %.1f min\n', N, t_min(end));
fprintf('固定分段点: t = %.1f min (索引 %d)\n', split_t_fixed, split_idx);
fprintf('总计拐点数: %d, 段数: %d\n', length(cp_idx), nSeg);
fprintf('拐点索引: %s\n', mat2str(cp_idx));
fprintf('拐点时间: %s min\n', mat2str(cp_time));
fprintf('各段多项式阶数: %s\n', mat2str(poly_order));

%% ---- 2. 分段拟合：以位移残差最小为目标求解速度多项式 ----
% 构建设计矩阵 A: x_model(t) = x_fact(t0) + A * p
% A(:,j) = (t^(ord+1-j) - t0^(ord+1-j)) / (ord+1-j)  逐次积分项
fprintf('\n======== 分段拟合结果与残差分析 ========\n');
v_fit = zeros(N, 1);
x_model = zeros(N, 1);
v_residual = zeros(N, 1);
p_seg = cell(nSeg, 1);   % 存储各段的多项式系数

for i = 1:nSeg
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end, continue; end
    
    seg_t = t_min(idx_start:idx_end);
    seg_x = x_fact(idx_start:idx_end);
    
    ord = poly_order(i);
    t0 = seg_t(1);
    x0 = seg_x(1);          % 段首位移
    
    % 构造积分设计矩阵 A
    % v(t) = p(1)*t^ord + p(2)*t^(ord-1) + ... + p(ord)*t + p(ord+1)
    % ∫v dt from t0 to t = Σ p(k) * (t^(ord+2-k) - t0^(ord+2-k)) / (ord+2-k)
    A = zeros(length(seg_t), ord+1);
    for k = 0:ord
        % 对应 p(ord+1-k) 的项：t^(k+1) 的积分
        A(:, ord+1-k) = (seg_t.^(k+1) - t0^(k+1)) / (k+1);
    end
    
    b = seg_x - x0;        % 目标：位移增量
    
    p = A \ b;             % 最小化 ||b - A*p||² 即位移残差
    p_seg{i} = p;
    
    % 计算速度拟合值
    seg_v_fit = polyval(p, seg_t);
    v_fit(idx_start:idx_end) = seg_v_fit;
    
    % 计算模型位移（解析积分）
    seg_x_model = x0 + A * p;
    x_model(idx_start:idx_end) = seg_x_model;
    
    % 速度残差（辅助参考）
    if idx_start <= size(v, 1) && idx_end <= size(v, 1)
        seg_v = v(idx_start:idx_end);
        seg_res_v = seg_v - seg_v_fit;
        rmse_v = sqrt(mean(seg_res_v.^2));
        mae_v  = mean(abs(seg_res_v));
        max_res_v = max(abs(seg_res_v));
        v_residual(idx_start:idx_end) = seg_res_v;
    end
    
    % 位移残差（主目标）
    seg_res_d = seg_x - seg_x_model;
    rmse_d = sqrt(mean(seg_res_d.^2));
    mae_d  = mean(abs(seg_res_d));
    max_d  = max(abs(seg_res_d));
    
    % 输出拟合公式
    fprintf('\n阶段 %d（t = %.1f ~ %.1f min，点 %d ~ %d，阶数 %d）:\n', ...
        i, seg_t(1), seg_t(end), idx_start, idx_end, ord);
    if ord == 1
        fprintf('  v(t) = %.6f + %.6e * t\n', p(2), p(1));
        fprintf('  等效加速度 = %.6e mm/min²\n', p(1));
    elseif ord == 2
        fprintf('  v(t) = %.6e * t^2 + %.6e * t + %.6f\n', p(1), p(2), p(3));
        fprintf('  a(t) = %.6e * t + %.6e\n', 2*p(1), p(2));
    else
        fprintf('  多项式系数: %s\n', mat2str(p));
    end
    fprintf('  位移 RMSE=%.6f mm, MAE=%.6f mm, Max=%.6f mm（以位移残差最小为目标）\n', ...
        rmse_d, mae_d, max_d);
    if idx_start <= size(v, 1) && idx_end <= size(v, 1)
        fprintf('  速度残差（辅助参考）: RMSE=%.6f, MAE=%.6f, Max=%.6f mm/min\n', rmse_v, mae_v, max_res_v);
    end
end

%% ---- 3. 计算位移残差（模型位移已在第2部分解析积分得到） ----
residual_disp = x_fact - x_model;

%% ---- 4a. 独立图：滤波后位移时序（区间背景着色） ----
% 区间背景色定义（第1段=浅绿，第2段=浅黄，第3/4段=浅红）
segBgColors = {[0.7 1 0.7], [1 1 0.2], [1 0.7 0.7], [1 0.7 0.7]};
segNames = {'第1段', '第2段', '第3段', '第4段'};
% 线条分段着色用颜色（第4部分使用）
segColors = {[0 0.6 0], [1 0.8 0.2], [0.8 0 0], [0.8 0 0], [0 0.3 0.8]};

figure('Position', [100, 100, 1000, 500]);
hold on;

% 先绘制位移曲线（统一用黑色）
plot(t_min, x_fact, 'k-', 'LineWidth', 1.2, 'DisplayName', '位移');
yl = ylim;

% 再绘制区间背景色块（基于已有 ylim）
for i = 1:min(nSeg, 4)   % 只处理前4段
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end, continue; end
    t_start = t_min(idx_start);
    t_end   = t_min(idx_end);
    patch([t_start, t_end, t_end, t_start], [yl(1), yl(1), yl(2), yl(2)], ...
        segBgColors{i}, 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
        'DisplayName', segNames{i});
end

% 绘制拐点竖线
for i = 1:length(cp_idx)
    xline(t_min(cp_idx(i)), '--k', 'LineWidth', 1.0, 'DisplayName', '拐点');
end

xlabel('时间 t (min)');
ylabel('位移 (mm)');
title('分段结果');
legend('位移','第1段','第2段','第3段','Location', 'best');
grid on;
hold off;

%% ---- 4. 绘图（按段着色） ----

figure('Position', [100, 100, 1400, 900]);

subplot(3,1,1);
hold on;
plot(t_min, v,'Color',[0.2 0.2 0.2],'LineWidth',2);
for i = 1:nSeg
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end, continue; end
    plot(t_min(idx_start:idx_end), v_fit(idx_start:idx_end), '-', ...
        'Color', segColors{min(i,5)}, 'LineWidth', 1.5);
end
for i = 1:length(cp_idx)
    xline(t_min(cp_idx(i)), '--', 'LineWidth', 1.0);
end
ylabel('速度 (mm/min)');
title('速度：测算速度 vs 多项式拟合速度');
legend('测算速度', '拟合速度（按段着色）', 'Location', 'best');
grid on;

subplot(3,1,2);
hold on;
plot(t_min, x_fact,'Color',[0.2 0.2 0.2],'LineWidth',2);
for i = 1:nSeg
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end, continue; end
    plot(t_min(idx_start:idx_end), x_model(idx_start:idx_end), '-', ...
        'Color', segColors{min(i,5)}, 'LineWidth', 1.5);
end
for i = 1:length(cp_idx)
    xline(t_min(cp_idx(i)), '--', 'LineWidth', 1.0);
end
ylabel('位移 (mm)');
title('位移：测算位移 vs 多项式拟合位移');
legend('测算位移', '拟合位移（按段着色）', 'Location', 'best');
grid on;

subplot(3,1,3);
hold on;
for i = 1:nSeg
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end, continue; end
    plot(t_min(idx_start:idx_end), residual_disp(idx_start:idx_end), '-', ...
        'Color', segColors{min(i,5)}, 'LineWidth', 0.8);
end
for i = 1:length(cp_idx)
    xline(t_min(cp_idx(i)), '--', 'LineWidth', 1.0);
end
xlabel('时间 t (min)');
ylabel('位移残差 (mm)');
title('位移拟合残差（实测 - 模型）');
grid on;

sgtitle(sprintf('混合分段模型验证（rms + linear）'));

%% ---- 5. 逐段位移残差分析 ----
fprintf('\n======== 各阶段位移拟合残差分析 ========\n');
for i = 1:nSeg
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start > idx_end, continue; end
    seg_res_disp = residual_disp(idx_start:idx_end);
    rmse_d = sqrt(mean(seg_res_disp.^2));
    mae_d  = mean(abs(seg_res_disp));
    max_d  = max(abs(seg_res_disp));
    fprintf('阶段 %d（t = %.1f ~ %.1f min）: 位移 RMSE=%.6f mm, MAE=%.6f mm, Max=%.6f mm\n', ...
        i, t_min(idx_start), t_min(idx_end), rmse_d, mae_d, max_d);
end

%% ---- 6. 整体评价 ----
SS_res = sum(residual_disp.^2);
SS_tot = sum((x_fact - mean(x_fact)).^2);
R_sq   = 1 - SS_res/SS_tot;
RMSE   = sqrt(mean(residual_disp.^2));
MAE    = mean(abs(residual_disp));

fprintf('\n======== 整体位移拟合评价 ========\n');
fprintf('决定系数 R²  : %.9f\n', R_sq);
fprintf('均方根误差 RMSE : %.4f mm\n', RMSE);
fprintf('平均绝对误差 MAE : %.4f mm\n', MAE);
fprintf('最大绝对残差     : %.4f mm\n', max(abs(residual_disp)));

if R_sq > 0.99999 && max(abs(residual_disp)) < 0.5
    fprintf('\n>>> 混合分段模型达到极高精度。\n');
else
    fprintf('\n>>> 残差仍有改善空间，可进一步微调阶数或分段点。\n');
end

%% ---- 6.5 前四段整体位移拟合评价 ----
% 取前 4 段对应的残差
first4_mask = false(N, 1);
for i = 1:min(4, nSeg)
    idx_start = segments(i, 1);
    idx_end   = segments(i, 2);
    if idx_start <= idx_end
        first4_mask(idx_start:idx_end) = true;
    end
end
residual_first4 = residual_disp(first4_mask);
x_fact_first4   = x_fact(first4_mask);

SS_res4 = sum(residual_first4.^2);
SS_tot4 = sum((x_fact_first4 - mean(x_fact_first4)).^2);
R_sq4   = 1 - SS_res4/SS_tot4;
RMSE4   = sqrt(mean(residual_first4.^2));
MAE4    = mean(abs(residual_first4));

fprintf('\n======== 前四段整体位移拟合评价 ========\n');
fprintf('决定系数 R²  : %.9f\n', R_sq4);
fprintf('均方根误差 RMSE : %.4f mm\n', RMSE4);
fprintf('平均绝对误差 MAE : %.4f mm\n', MAE4);
fprintf('最大绝对残差     : %.4f mm\n', max(abs(residual_first4)));

%% ---- 7. 保存 ----
saveas(gcf, 'Adaptive_Poly_Fit.png');
fprintf('\n图形已保存为 Adaptive_Poly_Fit.png\n');

%% ---- 8. 输出拟合位移序列到 xlsx ----
% 构建输出表格，格式与 Filtered_Result.csv 一致
% 将 FilteredDisplacement 替换为模型拟合位移 x_model
outputTable = table(t_idx, x_fact, x_model, ...
    'VariableNames', {'SerialNo', 'RawDisplacement', 'FilteredDisplacement'});

% 写入 xlsx 文件
outputFile = 'Assumption2_ModelDisplacement.xlsx';
writetable(outputTable, outputFile);
fprintf('\n拟合位移序列已保存至 %s\n', outputFile);
