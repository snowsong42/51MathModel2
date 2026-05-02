import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载与初步清洗 ====================
train_path = '附件4：监测数据（训练集与实验集）-问题4.xlsx'
test_path  = '附件4：监测数据（训练集与实验集）-问题4.xlsx'   # 实验集在同一文件

# 读取训练集和实验集
train_df = pd.read_excel(train_path, sheet_name='训练集')
test_df  = pd.read_excel(test_path, sheet_name='实验集')

# 查看列名和基本信息
print('训练集列名：', train_df.columns)
print('实验集列名：', test_df.columns)

# 统一列名（实验集多了一列“阶段标签”，训练集没有）
train_df = train_df.rename(columns={'表面位移_mm': '位移', '降雨量_mm': '降雨', 
                                     '孔隙水压力_kPa': '孔压', '微震事件数': '微震',
                                     '爆破点距离_m': '爆破距离', '单段最大药量_kg': '药量'})
test_df = test_df.rename(columns={'表面位移_mm': '位移', '降雨量_mm': '降雨', 
                                   '孔隙水压力_kPa': '孔压', '微震事件数': '微震',
                                   '爆破点距离_m': '爆破距离', '单段最大药量_kg': '药量'})

# 处理缺失值：爆破相关列空白填0，并新增“是否爆破”指示列
for df in [train_df, test_df]:
    df['爆破距离'].fillna(0, inplace=True)
    df['药量'].fillna(0, inplace=True)
    df['爆破发生'] = (df['爆破距离'] > 0).astype(int)

# 时间列转为 datetime 并排序
train_df['时间'] = pd.to_datetime(train_df['时间'])
test_df['时间']  = pd.to_datetime(test_df['时间'])
train_df = train_df.sort_values('时间').reset_index(drop=True)
test_df  = test_df.sort_values('时间').reset_index(drop=True)

print(f'训练集样本数: {len(train_df)}, 实验集样本数: {len(test_df)}')
print(train_df[['位移', '降雨', '孔压', '微震', '爆破距离', '药量', '爆破发生']].head())

# ==================== 2. 阶段自动划分（基于位移速率变点检测） ====================
def segment_by_rate(df, min_length=200, smooth_window=12):
    """
    使用位移梯度（差分）并结合滑动窗口统计，寻找加速段起点。
    返回每个样本的阶段标签 0,1,2。
    """
    # 计算位移梯度（10min差分），并平滑
    df = df.copy()
    df['梯度'] = df['位移'].diff().fillna(0)
    # 使用滑动窗口标准差反映波动
    df['梯度_std'] = df['梯度'].rolling(window=smooth_window, min_periods=1).std()
    # 计算梯度的累积和，判断整体趋势偏移
    grad_series = df['梯度'].values
    # 简单方法：根据位移绝对值和梯度大小划分
    # 阶段一：位移 < 5mm 且 梯度绝对值小
    # 阶段二：5 <= 位移 < 80 且 梯度增大
    # 阶段三：位移 >= 80 或 梯度急剧增大
    labels = np.zeros(len(df))
    for i in range(len(df)):
        disp = df.loc[i, '位移']
        grad = df.loc[i, '梯度']
        if disp < 5 and abs(grad) < 0.5:
            labels[i] = 1   # 阶段一
        elif disp < 80:
            labels[i] = 2   # 阶段二
        else:
            labels[i] = 3   # 阶段三

    # 进一步平滑：连续模式转换
    # 要求阶段标签变化不能过于频繁
    return labels

train_df['阶段'] = segment_by_rate(train_df)
print('阶段划分统计：\n', train_df['阶段'].value_counts())

# ==================== 3. 特征工程 ====================
def create_features(df, is_train=True):
    """
    构建时间滞后、滚动统计等特征，需保证不引入未来信息。
    """
    df = df.copy()
    # 滞后特征（过去1h, 3h, 6h, 12h, 24h 的窗口统计）
    lags = [6, 18, 36, 72, 144]   # 对应1h,3h,6h,12h,24h (10min间隔)
    for lag in lags:
        df[f'降雨_lag{lag}'] = df['降雨'].shift(lag).fillna(0)
        df[f'孔压_lag{lag}'] = df['孔压'].shift(lag).fillna(method='ffill')
        df[f'微震_lag{lag}'] = df['微震'].shift(lag).fillna(0)
    
    # 滚动窗口累计（过去3h、6h的降雨、微震总数）
    for win in [18, 36]:   # 3h, 6h
        df[f'降雨_win{win}_sum'] = df['降雨'].rolling(window=win, min_periods=1).sum()
        df[f'微震_win{win}_sum'] = df['微震'].rolling(window=win, min_periods=1).sum()
    
    # 孔压的变化率（一阶差分）
    df['孔压_diff'] = df['孔压'].diff().fillna(0)
    
    # 爆破的衰减记忆（指数加权）
    if is_train:
        # 构建爆破冲击衰减序列
        blast_effect = np.zeros(len(df))
        decay = 0.7  # 每小时衰减因子
        for i in range(1, len(df)):
            blast_effect[i] = blast_effect[i-1] * decay + df.loc[i, '药量']
        df['爆破衰减'] = blast_effect
    else:
        # 对测试集同样计算（注意需要序列从前向后计算）
        blast_effect = np.zeros(len(df))
        decay = 0.7
        for i in range(1, len(df)):
            blast_effect[i] = blast_effect[i-1] * decay + df.loc[i, '药量']
        df['爆破衰减'] = blast_effect

    # 周期性时间特征
    df['hour'] = df['时间'].dt.hour
    df['dayofweek'] = df['时间'].dt.dayofweek
    
    return df

train_feat = create_features(train_df, is_train=True)
test_feat  = create_features(test_df, is_train=False)

# 删除非必要列，定义特征列和target
exclude_cols = ['时间', '位移', '阶段', '阶段标签'] + ['爆破距离', '药量', '爆破发生', '梯度', '梯度_std']
feature_cols = [c for c in train_feat.columns if c not in exclude_cols]

print(f'特征列数: {len(feature_cols)}')
print('特征示例：', feature_cols[:15])

# ==================== 4. 分阶段建模与评估 ====================
# 对训练集划分阶段
train_data = train_feat.copy()
# 实验集已有“阶段标签”列，直接使用
test_data = test_feat.copy()
test_data['阶段'] = test_data['阶段标签']   # 1,2,3

# 存储各阶段模型和效果
models = {}
stage_scores = {}

# 针对每个阶段训练一个 LightGBM 回归模型
for stage in [1, 2, 3]:
    print(f'\n====== 阶段 {stage} ======')
    # 训练集
    stage_train = train_data[train_data['阶段'] == stage].copy()
    if len(stage_train) == 0:
        print(f'阶段 {stage} 无训练样本，跳过')
        continue
    X_train = stage_train[feature_cols]
    y_train = stage_train['位移']
    
    # 划分部分验证集查看效果
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    # 训练 LightGBM
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, 
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f'阶段 {stage} 验证集 RMSE: {rmse:.4f}, R2: {r2:.4f}')
    stage_scores[stage] = {'rmse': rmse, 'r2': r2}
    models[stage] = model
    
    # 绘制拟合效果图
    plt.figure(figsize=(10, 3))
    plt.plot(y_val.reset_index(drop=True), label='实际位移')
    plt.plot(pd.Series(y_pred, index=y_val.index).reset_index(drop=True), label='预测位移', alpha=0.8)
    plt.title(f'阶段 {stage} 位移预测验证')
    plt.legend()
    plt.show()

# ==================== 5. 对实验集进行预测 ====================
test_preds = np.zeros(len(test_data))
for stage in [1, 2, 3]:
    mask = test_data['阶段'] == stage
    if mask.sum() == 0:
        continue
    X_test_stage = test_data.loc[mask, feature_cols]
    # 确保列顺序一致
    X_test_stage = X_test_stage[feature_cols]
    pred_stage = models[stage].predict(X_test_stage)
    test_preds[mask] = pred_stage

test_data['预测位移'] = test_preds
print('\n实验集预测位移 前10行：')
print(test_data[['时间', '阶段', '预测位移']].head(10))

# ==================== 6. 输出结果 ====================
# 保存预测结果
result = test_data[['时间', '阶段标签', '预测位移']]
result.to_excel('实验集预测位移_问题4.1.xlsx', index=False)
print('预测结果已保存至 实验集预测位移_问题4.1.xlsx')