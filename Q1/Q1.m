clear; clc;
data = readtable('Filtered 1.xlsx', 'VariableNamingRule', 'preserve'); % 读取数据
col_names = data.Properties.VariableNames;
idx_A = find(contains(col_names, 'Data A') | contains(col_names, 'Data_A') | contains(col_names, 'Optical'), 1);
idx_B = find(contains(col_names, 'Data B') | contains(col_names, 'Data_B') | contains(col_names, 'Vibrating'), 1);
A = data{:, idx_A};
B = data{:, idx_B};
n = length(A); 
coeff = polyfit(B, A, 1); % 全样本线性拟合
alpha = coeff(1);
beta  = coeff(2);
fprintf('校正公式: A_corr = (A - %.4f)/%.4f\n', beta, alpha);
A_corr = (A - beta) / alpha;
res = B - A_corr;
MAE  = mean(abs(res));
RMSE = sqrt(mean(res.^2));
R2   = 1 - sum(res.^2)/sum((B-mean(B)).^2);
MBE  = mean(res);
fprintf('全样本: MAE=%.4f RMSE=%.4f R2=%.6f MBE=%.4f\n', MAE, RMSE, R2, MBE);
raw_delta = A - B; % 校正前后对比
mae_raw = mean(abs(raw_delta));
rmse_raw = sqrt(mean(raw_delta.^2));
fprintf('校正前 MAE=%.4f RMSE=%.4f | 降幅 MAE=%.2f%% RMSE=%.2f%%\n',...
    mae_raw, rmse_raw, (mae_raw-MAE)/mae_raw*100, (rmse_raw-RMSE)/rmse_raw*100);
x_verify = [7.132, 18.526, 84.337, 123.554, 167.667];
y_verify = (x_verify - beta)/alpha;
fprintf('表1.1 校正结果:\n');
for i=1:length(x_verify)
    fprintf('%12.3f -> %12.3f\n', x_verify(i), y_verify(i));
end
