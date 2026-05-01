%% 数据清洗脚本：Attachment 1 → Filtered 1
%  清洗规则：
%    1) A、B 各自拟合线性趋势，残差的 MAD 异常检测（阈值可调）
%    2) 差值 D = A - B，移动窗口局部 MAD 检测
%  输出：
%     - Filtered 1.xlsx           清洗后的数据
%     - 命令窗口：剔除规则、剔除明细、统计对比

clear; clc; close all;

%% ================= 参数设置 =================
THRESH_A_B = 100.0;          % A/B 单序列异常阈值（MAD 倍数）
WIN_D      = 500;          % D 序列移动窗口宽度（点数）
THRESH_D   = 22.0;         % D 序列 MAD 倍数阈值
DATA_FILE  = 'D:\project\pythonProject\MathModel\51MathModel2\Q1\Attachment 1.xlsx';
OUT_FILE   = 'Filtered 1.xlsx';
% =============================================

%% 1. 读取原始数据
fprintf('>>> 正在读取数据...\n');
raw = readtable(DATA_FILE, 'VariableNamingRule', 'preserve');
t_raw = datetime(raw.Time, 'InputFormat', 'yyyy-MM-dd HH:mm');
A_raw = raw.('Data A (Optical Fiber Displacement Sensor Data, mm)');
B_raw = raw.('Data B (Vibrating Wire Displacement Sensor Data, mm)');
n_raw = length(A_raw);
t_num = hours(t_raw - t_raw(1));   % 相对小时数，用于趋势拟合

%% 2. 第一步清洗：A、B 各自移除明显偏离整体线性趋势的点
fprintf('\n========== 第一步：A、B 单独清洗（线性去趋势 + MAD） ==========\n');

% 拟合线性趋势
pA = polyfit(t_num, A_raw, 1);
pB = polyfit(t_num, B_raw, 1);
trendA = polyval(pA, t_num);
trendB = polyval(pB, t_num);

% 残差
resA = A_raw - trendA;
resB = B_raw - trendB;

% 残差的 MAD（绝对中位差）
madA = median(abs(resA - median(resA)));
madB = median(abs(resB - median(resB)));

% 异常标记
outA = abs(resA) > THRESH_A_B * madA;
outB = abs(resB) > THRESH_A_B * madB;
out_AB = outA | outB;   % 合并标记：任一传感器异常即剔除整行

fprintf('A 残差 MAD = %.4f mm, 阈值 = %.4f mm\n', madA, THRESH_A_B*madA);
fprintf('B 残差 MAD = %.4f mm, 阈值 = %.4f mm\n', madB, THRESH_A_B*madB);
fprintf('第一步剔除 %d 个点（A异常:%d, B异常:%d, 合并:%d）\n',...
    sum(out_AB), sum(outA), sum(outB), sum(outA & outB));

% 记录被剔除的数据
removed1 = table(t_raw(out_AB), A_raw(out_AB), B_raw(out_AB),...
    'VariableNames', {'Time', 'A_removed', 'B_removed'});
if height(removed1) > 0
    fprintf('第一步剔除明细：\n');
    disp(removed1);
end

% 生成第一步清洗后的数据（去除异常行）
idx_keep1 = ~out_AB;
t1 = t_raw(idx_keep1);
A1 = A_raw(idx_keep1);
B1 = B_raw(idx_keep1);
n1 = length(A1);
fprintf('第一步清洗后剩余 %d 点。\n', n1);

%% 3. 第二步清洗：基于差值 D = A - B 的移动窗口 MAD 检测
fprintf('\n========== 第二步：差值 D 移动窗口 MAD 检测（窗宽 %d，阈值 %.1f×MAD） ==========\n',...
    WIN_D, THRESH_D);

D1 = A1 - B1;

% 移动窗口局部中位数与 MAD
local_median = nan(size(D1));
local_mad = nan(size(D1));

half_win = floor(WIN_D/2);
for i = 1:n1
    idx_start = max(1, i - half_win);
    idx_end = min(n1, i + half_win);
    win_D = D1(idx_start:idx_end);
    local_median(i) = median(win_D);
    local_mad(i) = median(abs(win_D - local_median(i)));
end

% 标准差估算：MAD → 标准差（若需要可选用 1.4826 倍，但题目直接使用 MAD）
% 此处严格按题目要求使用 MAD 本身（而非标准差），即直接比较 |D - 中位数| 与 25*MAD
out_D_idx = abs(D1 - local_median) > THRESH_D * local_mad;

% 补充：如果局部窗口内 MAD 为 0（所有值相同），则无法产生异常
out_D_idx(local_mad == 0) = false;

fprintf('第二步剔除 %d 个点。\n', sum(out_D_idx));

removed2 = table(t1(out_D_idx), A1(out_D_idx), B1(out_D_idx), D1(out_D_idx),...
    local_median(out_D_idx), local_mad(out_D_idx),...
    'VariableNames', {'Time', 'A_removed', 'B_removed', 'D_removed', 'LocalMedian', 'LocalMAD'});
if height(removed2) > 0
    fprintf('第二步剔除明细（部分显示）：\n');
    disp(removed2(1:min(10,height(removed2)),:));
end

% 最终清洗后数据
idx_keep2 = ~out_D_idx;
t_final = t1(idx_keep2);
A_final = A1(idx_keep2);
B_final = B1(idx_keep2);
D_final = A_final - B_final;
fprintf('最终清洗后剩余 %d 点。\n', length(A_final));

%% 4. 输出 Filtered 1.xlsx
fprintf('\n>>> 正在保存清洗后数据到 %s ...\n', OUT_FILE);
% 整理成表格
output_table = table(datestr(t_final, 'yyyy-mm-dd HH:MM'), A_final, B_final,...
    'VariableNames', {'Time', 'Data_A_Optical_mm', 'Data_B_Vibrating_mm'});
writetable(output_table, OUT_FILE);
fprintf('保存成功！\n');

%% 5. 可视化对比（清洗前后）
figure('Position', [100 100 1400 900]);

% --- 子图1：A 序列清洗前后对比 ---
subplot(3,2,1);
plot(t_raw, A_raw, 'Color', [0.7 0.7 0.7]); hold on;
plot(t_raw(out_AB), A_raw(out_AB), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(t_final, A_final, 'b-', 'LineWidth', 1);
title('A 序列清洗前后对比');
xlabel('时间'); ylabel('位移 (mm)');
legend('原始A', '第一步剔除', '清洗后A', 'Location','best'); grid on;

% --- 子图2：B 序列清洗前后对比 ---
subplot(3,2,2);
plot(t_raw, B_raw, 'Color', [0.7 0.7 0.7]); hold on;
plot(t_raw(out_AB), B_raw(out_AB), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(t_final, B_final, 'r-', 'LineWidth', 1);
title('B 序列清洗前后对比');
xlabel('时间'); ylabel('位移 (mm)');
legend('原始B', '第一步剔除', '清洗后B', 'Location','best'); grid on;

% --- 子图3：差值 D 序列清洗前后对比 ---
subplot(3,2,3);
plot(t_raw, A_raw - B_raw, 'Color', [0.7 0.7 0.7]); hold on;
% 标记两步分别剔除的点
plot(t_raw(out_AB), A_raw(out_AB) - B_raw(out_AB), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
% 第二步剔除（从t1中）
plot(t1(out_D_idx), D1(out_D_idx), 'mo', 'MarkerSize', 6, 'LineWidth', 1);
plot(t_final, D_final, 'k-', 'LineWidth', 1);
title('差值 D = A-B 清洗前后对比');
xlabel('时间'); ylabel('D (mm)');
legend('原始D', '第一步剔除', '第二步剔除', '清洗后D', 'Location','best'); grid on;

% --- 子图4：清洗后 A vs B 散点图 ---
subplot(3,2,4);
scatter(B_final, A_final, 8, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
limits_AB = [min([A_final;B_final]) max([A_final;B_final])];
plot(limits_AB, limits_AB, 'r--', 'LineWidth', 1.5);
p_final = polyfit(B_final, A_final, 1);
B_line = linspace(limits_AB(1), limits_AB(2), 100);
plot(B_line, polyval(p_final, B_line), 'g-', 'LineWidth', 2);
xlabel('B'); ylabel('A');
title(sprintf('清洗后 A vs B (拟合: A=%.3f·B+%.3f)', p_final(1), p_final(2)));
legend('数据点', 'y=x', '线性拟合', 'Location','best'); grid on;
axis equal;

% --- 子图5：移动窗口局部中位数及异常阈值 ---
subplot(3,2,[5,6]);
% 绘制局部中位数 ± 25*MAD 包络线与异常点
t1_num = hours(t1 - t1(1));  % 用于绘图
patch([t1_num; flipud(t1_num)], ...
      [local_median - THRESH_D*local_mad; flipud(local_median + THRESH_D*local_mad)], ...
      [0.9 0.9 0.9], 'EdgeColor', 'none'); hold on;
plot(t1_num, local_median, 'b-', 'LineWidth', 1.5);
plot(t1_num, D1, 'k.', 'MarkerSize', 4);
plot(t1_num(out_D_idx), D1(out_D_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
title(sprintf('移动窗口局部中位数 ± %.1f×MAD（窗宽 %d）', THRESH_D, WIN_D));
xlabel('相对时间 (小时)'); ylabel('D (mm)');
legend('异常阈值带', '局部中位数', 'D 序列', '第二步剔除点', 'Location','best');
grid on;

sgtitle('数据清洗效果总览', 'FontSize', 14);

%% 6. 清洗前后统计汇总
fprintf('\n================== 清洗前后统计对比 ==================\n');
fprintf('原始数据点数                : %d\n', n_raw);
fprintf('第一步（A/B单序列）剔除      : %d\n', sum(out_AB));
fprintf('第二步（D序列MAD）剔除       : %d\n', sum(out_D_idx));
fprintf('最终保留点数                 : %d\n', length(A_final));
fprintf('---------------------------------------------------\n');
fprintf('原始 A 均值 ± 标准差 : %8.4f ± %8.4f mm\n', mean(A_raw), std(A_raw));
fprintf('原始 B 均值 ± 标准差 : %8.4f ± %8.4f mm\n', mean(B_raw), std(B_raw));
fprintf('原始 D 均值 ± 标准差 : %8.4f ± %8.4f mm\n', mean(A_raw-B_raw), std(A_raw-B_raw));
fprintf('清洗后 A 均值 ± 标准差 : %8.4f ± %8.4f mm\n', mean(A_final), std(A_final));
fprintf('清洗后 B 均值 ± 标准差 : %8.4f ± %8.4f mm\n', mean(B_final), std(B_final));
fprintf('清洗后 D 均值 ± 标准差 : %8.4f ± %8.4f mm\n', mean(D_final), std(D_final));
fprintf('====================================================\n');

fprintf('\n脚本运行完毕。清洗后数据已保存至 %s\n', OUT_FILE);