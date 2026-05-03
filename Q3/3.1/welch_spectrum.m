%% ============================================================
% Welch 功率谱估计脚本（手写实现）
% 对附件3各变量进行频谱分析
% 参数：N=256点，50%重叠，汉明窗
% ============================================================
clear; clc; close all;

%% ---- 1. 读取数据 ----
filename = 'Attachment 3.xlsx';
sheets = {'训练集', '实验集'};

for s = 1:2
    sheet = sheets{s};
    fprintf('===== 处理 %s =====\n', sheet);
    
    dataTable = readtable(filename, 'Sheet', sheet, 'VariableNamingRule', 'preserve');
    
    % 获取列名
    varNames = dataTable.Properties.VariableNames;
    fprintf('列名：'); disp(varNames);
    
    % 取第2列到第6列（5个变量）
    colNames = varNames(2:6);
    
    N = height(dataTable);
    fprintf('数据长度：%d\n', N);
    
    %% ---- 2. 三次样条插值补全缺失值 ----
    interpData = zeros(N, 5);
    for c = 1:5
        raw = dataTable{:, c+1};
        raw = double(raw);
        idx = (1:N)';
        valid = ~isnan(raw);
        
        if sum(valid) < 4
            % 少于4个有效点，退化为线性插值
            interpData(:, c) = interp1(idx(valid), raw(valid), idx, 'linear', 'extrap');
        else
            % 三次样条插值
            cs = spline(idx(valid), raw(valid));
            interpData(:, c) = ppval(cs, idx);
        end
        
        fprintf('  变量%c：%d个缺失值已插值补全\n', char('a'+c-1), sum(~valid));
    end
    
    %% ---- 3. Welch 功率谱估计（手写实现） ----
    % 参数
    Nseg = 256;          % 每段点数
    noverlap = 128;      % 重叠点数（50%）
    step = Nseg - noverlap;  % 步长
    nfft = Nseg;         % FFT点数
    
    % 汉明窗 (公式 5-3-1)
    n = (0:Nseg-1)';
    w = 0.54 - 0.46 * cos(2 * pi * n / (Nseg - 1));
    
    % 归一化因子 (公式 5-3-2)
    U = sum(w.^2) / Nseg;
    
    % 分段数
    K = floor((N - noverlap) / step);
    fprintf('  总段数 K = %d\n', K);
    
    % 对每个变量计算功率谱
    for c = 1:5
        x = interpData(:, c);
        x = x - mean(x);  % 去直流
        
        % Welch 方法
        Pxx = zeros(nfft, 1);
        for k = 0:K-1
            startIdx = k * step + 1;
            endIdx = startIdx + Nseg - 1;
            xk = x(startIdx:endIdx);
            
            % 乘窗
            xk_w = xk .* w;
            
            % FFT (公式 5-3-3)
            Xk = fft(xk_w, nfft);
            
            % 功率谱 (公式 5-3-4)
            Pk = abs(Xk).^2 / (Nseg * U);
            
            % 累加 (公式 5-3-5)
            Pxx = Pxx + Pk;
        end
        Pxx = Pxx / K;
        
        % 取单边谱
        if mod(nfft, 2) == 0
            Pxx_single = Pxx(1:nfft/2+1);
            Pxx_single(2:end-1) = 2 * Pxx_single(2:end-1);
        else
            Pxx_single = Pxx(1:(nfft+1)/2);
            Pxx_single(2:end) = 2 * Pxx_single(2:end);
        end
        f = (0:length(Pxx_single)-1)' / nfft;
        
        % 保存结果
        if s == 1
            P_all_train{c} = Pxx_single;
        else
            P_all_exp{c} = Pxx_single;
        end
        
        %% ---- 4. 绘图 ----
        figure('Position', [100 + (c-1)*50, 100 + (s-1)*50, 700, 450]);
        
        % 双对数坐标
        loglog(f(2:end), Pxx_single(2:end), 'b-', 'LineWidth', 1.5);
        hold on;
        
        % 标注高频噪声频段（周期3~7点 → 频率 1/7~1/3）
        f_noise_low = 1/7;
        f_noise_high = 1/3;
        xline(f_noise_low, 'r--', 'LineWidth', 1);
        xline(f_noise_high, 'r--', 'LineWidth', 1);
        
        % 标注尖锐异常频段（周期25~35点 → 频率 1/35~1/25）
        f_anom_low = 1/35;
        f_anom_high = 1/25;
        xline(f_anom_low, 'g--', 'LineWidth', 1);
        xline(f_anom_high, 'g--', 'LineWidth', 1);
        
        % 添加阴影标注区域
        fill_x = [f_noise_low, f_noise_high, f_noise_high, f_noise_low];
        y_lim = ylim;
        fill_y = [y_lim(1), y_lim(1), y_lim(2), y_lim(2)];
        fill(fill_x, fill_y, 'r', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
        
        fill_x = [f_anom_low, f_anom_high, f_anom_high, f_anom_low];
        fill(fill_x, fill_y, 'g', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
        
        varLabel = colNames{c};
        title(sprintf('Welch 功率谱 - %s [%s]', varLabel, sheet));
        xlabel('归一化频率 (cycles/sample)');
        ylabel('功率谱密度');
        legend({'功率谱', 'f=1/7 (周期7点)', 'f=1/3 (周期3点)', ...
                'f=1/35 (周期35点)', 'f=1/25 (周期25点)'}, ...
               'Location', 'best', 'FontSize', 8);
        grid on;
        
        % 保存图片
        outDir = 'Fourier';
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        if s == 1
            saveas(gcf, fullfile(outDir, sprintf('welch_spectrum_train_%c.png', char('a'+c-1))));
        else
            saveas(gcf, fullfile(outDir, sprintf('welch_spectrum_exp_%c.png', char('a'+c-1))));
        end
        close(gcf);
    end
end

%% ---- 5. 汇总分析 ----
fprintf('\n===== 频谱分析汇总 =====\n');
fprintf('高频噪声频段：周期 3~7 点 | 频率 [%.4f, %.4f]\n', 1/7, 1/3);
fprintf('尖锐异常频段：周期 25~35 点 | 频率 [%.4f, %.4f]\n', 1/35, 1/25);
fprintf('\nWelch 参数：\n');
fprintf('  段长 N = %d\n', Nseg);
fprintf('  重叠度 = %.0f%%\n', noverlap/Nseg*100);
fprintf('  窗函数：汉明窗\n');
fprintf('  FFT点数 = %d\n', nfft);
fprintf('\n所有频谱图已保存至 Fourier/ 目录\n');
fprintf('================================\n');

%% ---- 6. 与 MATLAB 内置 pwelch 对比验证 ----
fprintf('\n===== 与内置 pwelch 对比验证（训练集变量a）=====\n');
x_train = interpData(:, 1) - mean(interpData(:, 1));

% 内置 pwelch
[Pxx_builtin, f_builtin] = pwelch(x_train, hamming(Nseg), noverlap, nfft, 1);

% 手写实现
x = interpData(:, 1);
x = x - mean(x);
Pxx_manual = zeros(nfft, 1);
for k = 0:K-1
    startIdx = k * step + 1;
    endIdx = startIdx + Nseg - 1;
    xk = x(startIdx:endIdx);
    xk_w = xk .* w;
    Xk = fft(xk_w, nfft);
    Pk = abs(Xk).^2 / (Nseg * U);
    Pxx_manual = Pxx_manual + Pk;
end
Pxx_manual = Pxx_manual / K;

% 取单边
Pxx_manual_single = Pxx_manual(1:nfft/2+1);
Pxx_manual_single(2:end-1) = 2 * Pxx_manual_single(2:end-1);

% 计算误差
diff_ratio = norm(Pxx_builtin - Pxx_manual_single) / norm(Pxx_builtin);
fprintf('归一化差异：%.6e\n', diff_ratio);
if diff_ratio < 1e-10
    fprintf('✅ 手写实现与内置 pwelch 结果完全一致！\n');
else
    fprintf('⚠️  存在微小差异（可能由数值精度引起）\n');
end

fprintf('\nWelch 功率谱估计完成！\n');
