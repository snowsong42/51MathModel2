clear; clc;
opts = detectImportOptions('Attachment 2.xlsx'); opts.VariableNamingRule = 'preserve';
data = readtable('Attachment 2.xlsx', opts);
raw = data{:,2}; N = length(raw); % 读取原始数据
zeroIdx = find(raw==0); % 零值插值
interpRaw = raw;
if ~isempty(zeroIdx)
    valid = find(raw~=0);
    interpRaw(zeroIdx) = interp1(valid, raw(valid), zeroIdx, 'linear','extrap');
end
med = medfilt1(interpRaw, 9); % 混合滤波
wav = wdenoise(med(:));
sg = sgolayfilt(wav, 2, 21);
dt = 10; [~,g] = sgolay(3,27); m = 13; % 计算速度 (sgolay微分)
v_raw = conv(sg, 1/(-dt)*g(:,2), 'same');
v_raw(1:m)=v_raw(m+1); v_raw(end-m+1:end)=v_raw(end-m);
% 速度混合滤波
v_filt = medfilt1(v_raw,100);
v_filt = wdenoise(v_filt(:));
v_filt = sgolayfilt(v_filt,2,45);
cp_idx = findchangepts(v_filt, 'MaxNumChanges',2, 'Statistic','rms'); % RMS突变点检测
t_min = (1:N)'*dt; cp_time = t_min(cp_idx);
seg = {1:cp_idx(1), cp_idx(1)+1:cp_idx(2), cp_idx(2)+1:N}; % 分段
ord = [1,2]; x_model = zeros(N,1); % 拟合与评估
idx=seg{1}; ti=t_min(idx); xi=sg(idx); t0=ti(1); x0=xi(1); % 第1段 1次多项式
A = [ti, ones(size(ti))]; p1 = A\(xi-x0); x_model(idx)=x0 + A*p1; 
idx=seg{2}; ti=t_min(idx); xi=sg(idx); t0=ti(1); x0=xi(1); % 第2段 2次多项式
A = [ti.^2, ti, ones(size(ti))]; p2 = A\(xi-x0); x_model(idx)=x0 + A*p2;
idx=seg{3}; ti=t_min(idx); xi=sg(idx); % 第3段 双指数
f = fit(ti, xi, fittype('exp2'), 'Algorithm','Levenberg-Marquardt');
x_model(idx) = feval(f, ti);
fprintf('拐点索引: %d, %d\n', cp_idx(1), cp_idx(2));
fprintf('拐点时间: %.1f min (%.2f 天), %.1f min (%.2f 天)\n',...
    cp_time(1), cp_time(1)/1440, cp_time(2), cp_time(2)/1440);
fprintf('阶段1 速度公式: v1(t)=%.4f mm/min (匀速)\n', p1(1));
fprintf('阶段2 速度公式: v2(t)=%.4e t^2 + %.4e t + %.4f\n', p2(1),p2(2),p2(3));
c = coeffvalues(f);
fprintf('阶段3 位移公式: x3(t)=%.4e e^{%.4f t} + %.4e e^{%.4f t}\n', c(1),c(2),c(3),c(4));
res = sg - x_model;
for k=1:3
    fprintf('阶段%d: RMSE=%.4f mm, MAE=%.4f mm\n', k, sqrt(mean(res(seg{k}).^2)), mean(abs(res(seg{k}))));
end
fprintf('整体 R²=%.6f, RMSE=%.4f mm\n', 1-sum(res.^2)/sum((sg-mean(sg)).^2), sqrt(mean(res.^2)));


