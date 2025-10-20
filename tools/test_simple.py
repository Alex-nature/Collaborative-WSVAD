"""简单测试：验证compute_change_rate函数的修复"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.adaptive_temporal import compute_change_rate

print("=" * 60)
print("测试 compute_change_rate 函数")
print("=" * 60)

# 测试案例1: [B, T, D] 格式 (batch小, 时序长, 特征维大)
print("\n测试1: [B, T, D] 格式 - batch=4, seq=256, dim=512")
x1 = torch.randn(4, 256, 512)
delta1 = compute_change_rate(x1)
print(f"输入: {x1.shape} -> 输出: {delta1.shape}")
print(f"预期: torch.Size([4, 256])")
assert delta1.shape == torch.Size([4, 256]), f"错误！实际形状: {delta1.shape}"
print("✓ 通过")

# 测试案例2: [T, B, D] 格式 (时序长, batch小, 特征维大)
print("\n测试2: [T, B, D] 格式 - seq=256, batch=4, dim=512")
x2 = torch.randn(256, 4, 512)
delta2 = compute_change_rate(x2)
print(f"输入: {x2.shape} -> 输出: {delta2.shape}")
print(f"预期: torch.Size([4, 256])")
assert delta2.shape == torch.Size([4, 256]), f"错误！实际形状: {delta2.shape}"
print("✓ 通过")

# 测试案例3: 小序列 [B, T, D]
print("\n测试3: [B, T, D] 格式 - batch=8, seq=32, dim=512")
x3 = torch.randn(8, 32, 512)
delta3 = compute_change_rate(x3)
print(f"输入: {x3.shape} -> 输出: {delta3.shape}")
print(f"预期: torch.Size([8, 32])")
assert delta3.shape == torch.Size([8, 32]), f"错误！实际形状: {delta3.shape}"
print("✓ 通过")

# 测试案例4: [T, B, D] 小序列
print("\n测试4: [T, B, D] 格式 - seq=32, batch=2, dim=512")
x4 = torch.randn(32, 2, 512)
delta4 = compute_change_rate(x4)
print(f"输入: {x4.shape} -> 输出: {delta4.shape}")
print(f"预期: torch.Size([2, 32])")
assert delta4.shape == torch.Size([2, 32]), f"错误！实际形状: {delta4.shape}"
print("✓ 通过")

# 测试案例5: 边界情况 - batch > seq
print("\n测试5: [B, T, D] 格式 - batch=16, seq=8, dim=512")
x5 = torch.randn(16, 8, 512)
delta5 = compute_change_rate(x5)
print(f"输入: {x5.shape} -> 输出: {delta5.shape}")
print(f"预期: torch.Size([16, 8])")
assert delta5.shape == torch.Size([16, 8]), f"错误！实际形状: {delta5.shape}"
print("✓ 通过")

print("\n" + "=" * 60)
print("✓ 所有测试通过！compute_change_rate函数工作正常")
print("=" * 60)

