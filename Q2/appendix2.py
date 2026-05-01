import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']   #  Windows/Mac 常见中文字体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 读取附件2数据
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Attachment 2.xlsx")
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 查看列名（实际应为 'Serial No.' 和 'Surface Displacement (mm)'）
print("列名：", df.columns.tolist())
# 通常第一列是序号，第二列是位移
x = df.iloc[:, 0]   # 序号
y = df.iloc[:, 1]   # 表面位移 (mm)

# 创建图形
plt.figure(figsize=(14, 6))
plt.plot(x, y, linewidth=0.8, color='steelblue', label='表面位移')

# 可选：标记部分噪声点（如大于3倍局部均值的点），但不强求，仅作示意
# 计算简单的滑动窗口均值与标准差，标出明显跳变点（可根据需要添加）
threshold = y.mean() + 3 * y.std()
outliers = y[y > threshold]
plt.scatter(outliers.index, outliers, color='red', s=20, alpha=0.6, label='异常跳变候选')

plt.xlabel('采集序号', fontsize=12)
plt.ylabel('表面位移 (mm)', fontsize=12)
plt.title('边坡表面位移时序曲线 (附件2)', fontsize=14)
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# 如果需要保存图片
# plt.savefig('displacement_curve.png', dpi=300, bbox_inches='tight')
