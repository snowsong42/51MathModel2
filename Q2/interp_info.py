import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('Q2/Attachment 2.xlsx')
serial_no = df.iloc[:, 0].values
raw_disp = df.iloc[:, 1].values

# 找到零值点
zero_idx = np.where(raw_disp == 0)[0]  # 0-based index
valid_idx = np.where(raw_disp != 0)[0]

print("=" * 70)
print("零值线性插值详细信息")
print("=" * 70)
print(f"\n数据总长度: {len(raw_disp)}")
print(f"零值点个数: {len(zero_idx)}")
print(f"零值点序号(Serial No.): {[serial_no[i] for i in zero_idx]}")
print()

# 对每个零值点进行插值
interp_values = {}
for zi in zero_idx:
    serial = serial_no[zi]
    raw_val = raw_disp[zi]
    
    # 线性插值（使用所有有效点，但这里只显示局部信息）
    interp_val = np.interp(zi, valid_idx, raw_disp[valid_idx])
    
    # 找前后相邻有效点
    left_valid = valid_idx[valid_idx < zi]
    right_valid = valid_idx[valid_idx > zi]
    
    left_idx = left_valid[-1] if len(left_valid) > 0 else None
    right_idx = right_valid[0] if len(right_valid) > 0 else None
    
    print(f"▶ 序号 {serial}:")
    print(f"   原始值 = {raw_val}")
    if left_idx is not None and right_idx is not None:
        left_serial = serial_no[left_idx]
        right_serial = serial_no[right_idx]
        left_val = raw_disp[left_idx]
        right_val = raw_disp[right_idx]
        
        # 线性插值计算
        t = (zi - left_idx) / (right_idx - left_idx)
        manual_interp = left_val + t * (right_val - left_val)
        
        print(f"   前序 {left_serial} (idx {left_idx+1}): {left_val}")
        print(f"   后序 {right_serial} (idx {right_idx+1}): {right_val}")
        print(f"   插值结果 = {interp_val:.6f}")
        print(f"   验证: {left_val} + ({zi+1}-{left_serial})/({right_serial}-{left_serial}) × ({right_val}-{left_val})")
        print(f"       = {left_val} + {t:.6f} × {right_val - left_val:.6f}")
        print(f"       = {manual_interp:.6f}")
    elif left_idx is None:
        print(f"   左侧无有效点（使用外推）")
        print(f"   插值结果 = {interp_val:.6f}")
    elif right_idx is None:
        print(f"   右侧无有效点（使用外推）")
        print(f"   插值结果 = {interp_val:.6f}")
    print()
    
    interp_values[serial] = interp_val

print("=" * 70)
print("插值结果汇总")
print("=" * 70)
print(f"{'序号':>8} {'原始值':>10} {'插值后':>12}")
print("-" * 32)
for serial in sorted(interp_values.keys()):
    print(f"{serial:>8} {'0.000':>10} {interp_values[serial]:>12.6f}")
