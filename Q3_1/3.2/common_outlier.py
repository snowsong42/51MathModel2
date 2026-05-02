"""
表3.2 共同异常点统计汇总
读取 common_outliers_table3.2.xlsx，按组合类型计数并排序输出
同时将汇总结果保存为 .xlsx 文件
"""
import pandas as pd
import os

# 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
xlsx_path = os.path.join(script_dir, "common_outliers_table3.2.xlsx")

# 读取
df = pd.read_excel(xlsx_path)

# 对 Common Abnormals 列计数，从高到低排序
counts = df['Common Abnormals'].value_counts().sort_values(ascending=False)

# 变量名称映射
var_names = {
    'a': '降雨量', 'b': '孔隙水压力', 'c': '微震事件数',
    'd': '深部位移', 'e': '表面位移'
}

# ===== 输出到控制台 =====
print("=" * 60)
print("共同异常点组合类型统计汇总")
print("=" * 60)
print(f"{'组合类型':<10} {'涉及变量':<35} {'出现次数':>8}  {'占比':>8}")
print("-" * 65)

total = counts.sum()
for combo, cnt in counts.items():
    vars_meaning = ' + '.join(var_names[ch] for ch in combo if ch in var_names)
    pct = cnt / total * 100
    print(f"{combo:<10} {vars_meaning:<35} {cnt:>8}  {pct:>7.2f}%")

print("-" * 65)
print(f"{'总计':<10} {'':<35} {total:>8}  {100.00:>7.2f}%")
print("=" * 60)

# 额外统计：涉及变量数量分组
grouped_by_count = {}
for combo, cnt in counts.items():
    n_vars = len(combo)
    grouped_by_count[n_vars] = grouped_by_count.get(n_vars, 0) + cnt

print(f"\n按涉及变量数量分组：")
for n_vars in sorted(grouped_by_count.keys()):
    print(f"  {n_vars}个变量同时异常: {grouped_by_count[n_vars]} 个时间点 ({grouped_by_count[n_vars]/total*100:.2f}%)")

# ===== 保存到 .xlsx =====
# 表1：按组合类型汇总
rows = []
for combo, cnt in counts.items():
    vars_meaning = ' + '.join(var_names[ch] for ch in combo if ch in var_names)
    rows.append({
        '组合类型': combo,
        '涉及变量': vars_meaning,
        '出现次数': cnt,
        '占比(%)': round(cnt / total * 100, 2)
    })
rows.append({
    '组合类型': '总计',
    '涉及变量': '',
    '出现次数': total,
    '占比(%)': 100.00
})
df_detail = pd.DataFrame(rows)

# 表2：按涉及变量数量分组
group_rows = []
for n_vars in sorted(grouped_by_count.keys()):
    group_rows.append({
        '同时异常变量数': f'{n_vars}个',
        '时间点数量': grouped_by_count[n_vars],
        '占比(%)': round(grouped_by_count[n_vars] / total * 100, 2)
})
df_group = pd.DataFrame(group_rows)

# 写入 Excel
output_path = os.path.join(script_dir, "common_outlier_summary.xlsx")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_detail.to_excel(writer, sheet_name='按组合类型汇总', index=False)
    df_group.to_excel(writer, sheet_name='按变量数量分组', index=False)
print(f"\n汇总结果已保存至 {output_path}")
