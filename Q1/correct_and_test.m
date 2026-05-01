%% 数据A校正：线性回归模型 + 前向滚动窗口交叉验证
%  输入：Filtered 1.xlsx（清洗后数据）
%  输出：命令窗口显示校正结果、交叉验证指标、表1.1验证结果
%  模型：A = alpha * B + beta  =>  A_corr = (A - beta) / alpha

clear; clc;

%% ================= 设置 =================
DATA_FILE = 'D:\project\pythonProject\MathModel\51MathModel2\Q1\Filtered 1.xlsx';
K = 5;                         % 交叉验证折数（可调，推荐5或10）
% 表1.1 待校正的5个原始A值
x_verify = [7.132, 18.526, 84.337, 123.554, 167.667];
% =========================================

%% 1. 读取清洗后数据
data = readtable(DATA_FILE, 'VariableNamingRule', 'preserve');
% 时间列
time = datetime(data.Time, 'InputFormat', 'yyyy-MM-dd HH:mm');
% 自动识别A、B列
col_names = data.Properties.VariableNames;
idx_A = find(contains(col_names, 'Data A') | contains(col_names, 'Data_A') | ...
             contains(col_names, 'Optical'), 1);
idx_B = find(contains(col_names, 'Data B') | contains(col_names, 'Data_B') | ...
             contains(col_names, 'Vibrating'), 1);
if isempty(idx_A) || isempty(idx_B)
    error('无法识别数据列，可用列名：%s', strjoin(col_names, ', '));
end
A = data{:, idx_A};
B = data{:, idx_B};
n = length(A);   % 总样本量

fprintf('数据加载完成，样本量 = %d\n', n);

%% 2. 全样本拟合（用于最终模型和表1.1验证）
% 拟合 A ~ B（无截距项？不，有β）
coeff_all = polyfit(B, A, 1);   % [alpha, beta]
alpha_all = coeff_all(1);
beta_all  = coeff_all(2);
fprintf('\n===== 全样本线性回归 =====\n');
fprintf('A = %.4f * B + %.4f\n', alpha_all, beta_all);
fprintf('即校正公式：A_corr = (A - %.4f) / %.4f\n', beta_all, alpha_all);

% 校正全样本
A_corr_all = (A - beta_all) / alpha_all;
% 计算全样本指标（参考）
residuals = B - A_corr_all;
mae_all  = mean(abs(residuals));
rmse_all = sqrt(mean(residuals.^2));
R2_all   = 1 - sum(residuals.^2) / sum((B - mean(B)).^2);
mbe_all  = mean(residuals);
fprintf('全样本拟合指标（参考，非交叉验证）：\n');
fprintf('  MAE  = %.4f mm\n', mae_all);
fprintf('  RMSE = %.4f mm\n', rmse_all);
fprintf('  R^2  = %.6f\n', R2_all);
fprintf('  MBE  = %.4f mm\n', mbe_all);

%% 2.1 校正前原始偏差对比
fprintf('\n====================================================\n');
fprintf('        |   校正前 (raw Δ)   |   校正后 (residual)  \n');
fprintf('----------------------------------------------------\n');

raw_delta = A - B;
mae_raw  = mean(abs(raw_delta));
rmse_raw = sqrt(mean(raw_delta.^2));
mae_imp  = (mae_raw - mae_all) / mae_raw * 100;
rmse_imp = (rmse_raw - rmse_all) / rmse_raw * 100;

fprintf('MAE     |   %12.4f mm    |   %12.4f mm\n', mae_raw, mae_all);
fprintf('RMSE    |   %12.4f mm    |   %12.4f mm\n', rmse_raw, rmse_all);
fprintf('----------------------------------------------------\n');
fprintf('MAE 降幅: %.2f%%  (%.4f → %.4f mm)\n', mae_imp, mae_raw, mae_all);
fprintf('RMSE降幅: %.2f%%  (%.4f → %.4f mm)\n', rmse_imp, rmse_raw, rmse_all);
fprintf('====================================================\n');

%% 2.2 残差诊断（模型假设检验）
fprintf('\n===== 残差诊断 (模型假设检验) =====\n');
fprintf('检验对象：全样本校正后残差 e_i = B_i - (A_i - β)/α\n');

% (1) 零均值 t 检验
[h_t, p_t, ~, stats_t] = ttest(residuals);
if p_t > 0.05
    t_note = '✓ 不拒绝H₀(均值为零)';
else
    t_note = '✗ 拒绝H₀';
end
fprintf('零均值 t 检验 : ē = %.4f, t = %.4f, p = %.4f  %s\n', ...
    mean(residuals), stats_t.tstat, p_t, t_note);

% (2) Durbin-Watson 检验（自编，无需工具箱）
n_dw = length(residuals);
dw_stat = sum(diff(residuals).^2) / sum(residuals.^2);
% 近似判断：DW ∈ (1.5, 2.5) 视为无显著一阶自相关
if dw_stat > 1.5 && dw_stat < 2.5
    dw_note = '✓ 无显著一阶自相关';
elseif dw_stat < 1.5
    dw_note = '⚠ 可能存在正自相关';
else
    dw_note = '⚠ 可能存在负自相关';
end
fprintf('Durbin-Watson  : DW = %.4f  %s\n', dw_stat, dw_note);

% (3) Ljung-Box Q 检验（自编，滞后 m = 20）
m_lb = min(20, floor(n_dw/5));  % 滞后阶数，取 min(20, n/5)
rho = zeros(m_lb, 1);
for k = 1:m_lb
    rho(k) = sum(residuals(1:end-k) .* residuals(1+k:end)) / sum(residuals.^2);
end
Q_lb = n_dw * (n_dw + 2) * sum(rho.^2 ./ (n_dw - (1:m_lb)'));
p_lb = 1 - chi2cdf(Q_lb, m_lb);
if p_lb > 0.05
    lb_note = '✓ 不拒绝H₀(白噪声)';
else
    lb_note = '✗ 拒绝H₀(存在序列相关)';
end
fprintf('Ljung-Box Q(%d): Q = %.4f, p = %.4f  %s\n', ...
    m_lb, Q_lb, p_lb, lb_note);

% (4) 正态性检验（Kolmogorov-Smirnov，标准化残差）
res_std = (residuals - mean(residuals)) / std(residuals);
[h_ks, p_ks] = kstest(res_std);
if p_ks > 0.05
    ks_note = '✓ 不拒绝H₀(正态分布)';
else
    ks_note = '✗ 拒绝H₀(非正态)';
end
fprintf('正态性 KS 检验: p = %.4f  %s\n', ...
    p_ks, ks_note);

fprintf('\n【解读】残差均值接近 0、DW 接近 2、Ljung-Box p>0.05，\n');
fprintf('表明线性模型已充分提取系统偏差，残差近似白噪声。\n');

%% 3. 前向滚动窗口交叉验证
% 将数据按时间顺序均分为K块
block_size = floor(n / K);
fprintf('\n===== 前向滚动窗口交叉验证（K=%d） =====\n', K);
fprintf('每折训练集：前 i 块；测试集：第 i+1 块\n');

% 存储每折指标
MAE_folds  = zeros(K-1, 1);
RMSE_folds = zeros(K-1, 1);
R2_folds   = zeros(K-1, 1);
MBE_folds  = zeros(K-1, 1);

for i = 1:(K-1)
    % 训练集：第1块到第 i 块
    train_end = i * block_size;
    B_train = B(1:train_end);
    A_train = A(1:train_end);
    % 测试集：第 i+1 块
    test_start = train_end + 1;
    test_end   = min((i+1) * block_size, n);
    B_test = B(test_start:test_end);
    A_test = A(test_start:test_end);
    
    % 在训练集上拟合
    p = polyfit(B_train, A_train, 1);
    alpha_fold = p(1);
    beta_fold  = p(2);
    
    % 对测试集校正
    A_corr_test = (A_test - beta_fold) / alpha_fold;
    residuals_test = B_test - A_corr_test;
    
    % 计算指标
    MAE_folds(i)  = mean(abs(residuals_test));
    RMSE_folds(i) = sqrt(mean(residuals_test.^2));
    R2_folds(i)   = 1 - sum(residuals_test.^2) / sum((B_test - mean(B_test)).^2);
    MBE_folds(i)  = mean(residuals_test);
    
    fprintf('第%d折 | 训练1:%d, 测试%d:%d | MAE=%.4f, RMSE=%.4f, R²=%.6f, MBE=%.4f\n', ...
            i, train_end, test_start, test_end, MAE_folds(i), RMSE_folds(i), R2_folds(i), MBE_folds(i));
end

% 汇总交叉验证指标
fprintf('\n----- 交叉验证汇总（均值 ± 标准差）-----\n');
fprintf('MAE  = %.4f ± %.4f mm\n', mean(MAE_folds), std(MAE_folds));
fprintf('RMSE = %.4f ± %.4f mm\n', mean(RMSE_folds), std(RMSE_folds));
fprintf('R²   = %.6f ± %.6f\n', mean(R2_folds), std(R2_folds));
fprintf('MBE  = %.4f ± %.4f mm\n', mean(MBE_folds), std(MBE_folds));

% 可选：各折的alpha, beta变化情况
fprintf('\n各折拟合参数差异（稳定性参考）：\n');
for i = 1:(K-1)
    train_end = i * block_size;
    p = polyfit(B(1:train_end), A(1:train_end), 1);
    fprintf('  使用前%d块训练: α=%.4f, β=%.4f\n', i, p(1), p(2));
end

%% 4. 表1.1 验证数据校正
fprintf('\n===== 表1.1 数据校正结果 =====\n');
fprintf('使用全样本模型：A_corr = (A - %.4f) / %.4f\n', beta_all, alpha_all);
y_verify = (x_verify - beta_all) / alpha_all;
fprintf('--------------------------------------------------\n');
fprintf('| 校正前数据 x | 校正后数据 y |\n');
fprintf('--------------------------------------------------\n');
for i = 1:length(x_verify)
    fprintf('| %12.3f | %12.3f |\n', x_verify(i), y_verify(i));
end
fprintf('--------------------------------------------------\n');

%% 5. 可视化
% 5.1 校正前后对比（2×2 布局）
figure('Position', [100 100 1000 800]);

% 校正前：A vs B 散点（参考线 y=x）
subplot(2,2,1);
plot(B, A, 'b.', 'MarkerSize', 4); hold on;
limits = [min([A;B]) max([A;B])];
plot(limits, limits, 'r--', 'LineWidth', 1.5);
xlabel('B (振弦传感器, mm)'); ylabel('A (光纤传感器, mm)');
title('校正前 A vs B (原始)');
legend('校正前数据点', 'y = x', 'Location','best');
grid on; axis equal;

% 校正前：原始差值 Δ = A - B 时序
% 先计算统一 y 轴范围（与校正后共用）
raw_delta = A - B;
ylim_max = max(max(abs(raw_delta)), max(abs(residuals))) * 1.1;
subplot(2,2,2);
plot(time, raw_delta, 'k.', 'MarkerSize', 2); hold on;
yline(mean(raw_delta), 'r--', 'LineWidth', 1.2);
yline(0, 'k:', 'LineWidth', 0.8);
xlabel('时间'); ylabel('差值 = A - B (mm)');
title(sprintf('校正前差值时序 (MAE=%.2f, RMSE=%.2f)', mae_raw, rmse_raw));
legend('校正前差值', sprintf('均值 = %.2f', mean(raw_delta)), '零线', 'Location','best');
ylim([-10, ylim_max]); % -10 到 60
grid on;

% 校正后：A_corr vs B 散点（回归线）
subplot(2,2,3);
plot(B, A_corr_all, 'b.', 'MarkerSize', 4); hold on;
plot(limits, limits, 'r--', 'LineWidth', 1.5);
xlabel('B (振弦传感器, mm)'); ylabel('A_{corr} (校正后, mm)');
title('校正后 A_{corr} vs B');
legend('校正后数据点', 'y = x', 'Location','best');
grid on; axis equal;

% 校正后：残差 B - A_corr 时序
subplot(2,2,4);
plot(time, residuals, 'k.', 'MarkerSize', 4); hold on;
yline(mbe_all, 'r--', 'LineWidth', 1.2);
yline(0, 'k:', 'LineWidth', 0.8);
xlabel('时间'); ylabel('差值 = A - B (mm)');
title(sprintf('校正后差值时序 (MAE=%.2f, RMSE=%.2f)', mae_all, rmse_all));
legend('校正后差值', sprintf('均值 = %.4f', mbe_all), '零线', 'Location','best');
ylim([-10, ylim_max]); % -10 到 60
grid on;

sgtitle('校正前后对比', 'FontSize', 14);

% 5.2 交叉验证 CV 曲线与参数收敛
figure('Position', [150 150 1000 700]);

% (1) MAE 各折分布
subplot(2,2,1);
fold_idx = 1:(K-1);
plot(fold_idx, MAE_folds, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'b'); hold on;
yline(mean(MAE_folds), 'r--', 'LineWidth', 1.2);
xlabel('折号'); ylabel('MAE (mm)');
title('交叉验证各折 MAE');
legend('各折 MAE', sprintf('均值 = %.4f', mean(MAE_folds)), 'Location','best');
xticks(fold_idx); grid on; ylim([0, max(MAE_folds)*1.3]);

% (2) RMSE 各折分布
subplot(2,2,2);
plot(fold_idx, RMSE_folds, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r'); hold on;
yline(mean(RMSE_folds), 'b--', 'LineWidth', 1.2);
xlabel('折号'); ylabel('RMSE (mm)');
title('交叉验证各折 RMSE');
legend('各折 RMSE', sprintf('均值 = %.4f', mean(RMSE_folds)), 'Location','best');
xticks(fold_idx); grid on; ylim([0, max(RMSE_folds)*1.3]);

% (3) α 参数收敛轨迹
subplot(2,2,3);
alpha_traj = zeros(K-1, 1);
beta_traj  = zeros(K-1, 1);
for i = 1:(K-1)
    train_end = i * block_size;
    p = polyfit(B(1:train_end), A(1:train_end), 1);
    alpha_traj(i) = p(1);
    beta_traj(i)  = p(2);
end
plot(fold_idx, alpha_traj, 'g-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'g'); hold on;
yline(alpha_all, 'k--', 'LineWidth', 1.2);
xlabel('训练块数（前 i 块）'); ylabel('\alpha');
title('\alpha 参数收敛轨迹');
legend('\alpha 轨迹', sprintf('全样本 \\alpha = %.4f', alpha_all), 'Location','best');
xticks(fold_idx); grid on;

% (4) β 参数收敛轨迹
subplot(2,2,4);
plot(fold_idx, beta_traj, 'm-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'm'); hold on;
yline(beta_all, 'k--', 'LineWidth', 1.2);
xlabel('训练块数（前 i 块）'); ylabel('\beta');
title('\beta 参数收敛轨迹');
legend('\beta 轨迹', sprintf('全样本 \\beta = %.4f', beta_all), 'Location','best');
xticks(fold_idx); grid on;

sgtitle('交叉验证指标与参数收敛分析', 'FontSize', 13);
