"""
测试自适应时序建模模块的维度兼容性
验证与原有模型的输入输出一致性
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.adaptive_temporal import AdaptiveConvGCNTemporal
from utils.hierarchical_transformer import HierarchicalTransformer
from utils.model import TransformerEncoder

def test_dimension_compatibility():
    """测试维度兼容性"""
    print("=" * 60)
    print("测试自适应时序建模模块维度兼容性")
    print("=" * 60)
    
    # 测试参数（与CLIP ViT-B/16匹配）
    batch_size = 4
    seq_len = 256  # visual_length
    feature_dim = 512  # visual_width
    
    # 创建测试输入 [T, B, D] (原始模型格式)
    x_tbd = torch.randn(seq_len, batch_size, feature_dim)
    print(f"\n✓ 输入形状 [T, B, D]: {x_tbd.shape}")
    
    # 测试1: 自适应Conv+GCN模块
    print("\n【测试1】自适应Conv+GCN模块")
    print("-" * 60)
    adaptive_model = AdaptiveConvGCNTemporal(
        width=feature_dim,
        gcn_layers=2,
        min_window=2,
        max_window=16,
        dropout=0.1,
        use_feature_sim=True,
        weight_hidden_dim=128
    )
    
    with torch.no_grad():
        out_adaptive = adaptive_model(x_tbd)
    
    print(f"输入形状: {x_tbd.shape}")
    print(f"输出形状: {out_adaptive.shape}")
    print(f"维度匹配: {out_adaptive.shape == x_tbd.shape}")
    print(f"特征维度保持: {out_adaptive.shape[-1] == feature_dim}")
    assert out_adaptive.shape == x_tbd.shape, "自适应模块输出维度不匹配！"
    print("✓ 自适应Conv+GCN模块测试通过")
    
    # 测试2: 对比HierarchicalTransformer
    print("\n【测试2】对比HierarchicalTransformer")
    print("-" * 60)
    hierarchical_model = HierarchicalTransformer(
        width=feature_dim,
        local_layers=2,
        global_layers=2,
        heads=8,
        window_size=16,
        dropout=0.1
    )
    
    with torch.no_grad():
        out_hierarchical = hierarchical_model(x_tbd)
    
    print(f"HierarchicalTransformer输出形状: {out_hierarchical.shape}")
    print(f"AdaptiveConvGCN输出形状: {out_adaptive.shape}")
    print(f"两者形状一致: {out_hierarchical.shape == out_adaptive.shape}")
    assert out_hierarchical.shape == out_adaptive.shape, "与HierarchicalTransformer输出维度不一致！"
    print("✓ 与HierarchicalTransformer兼容")
    
    # 测试3: 对比TransformerEncoder
    print("\n【测试3】对比TransformerEncoder")
    print("-" * 60)
    transformer_model = TransformerEncoder(
        width=feature_dim,
        layers=2,
        heads=8,
        dropout=0.1
    )
    
    with torch.no_grad():
        out_transformer = transformer_model(x_tbd)
    
    print(f"TransformerEncoder输出形状: {out_transformer.shape}")
    print(f"AdaptiveConvGCN输出形状: {out_adaptive.shape}")
    print(f"两者形状一致: {out_transformer.shape == out_adaptive.shape}")
    assert out_transformer.shape == out_adaptive.shape, "与TransformerEncoder输出维度不一致！"
    print("✓ 与TransformerEncoder兼容")
    
    # 测试4: [B, T, D]格式输入
    print("\n【测试4】[B, T, D]格式兼容性")
    print("-" * 60)
    x_btd = torch.randn(batch_size, seq_len, feature_dim)
    
    with torch.no_grad():
        out_btd = adaptive_model(x_btd)
    
    print(f"输入形状: {x_btd.shape}")
    print(f"输出形状: {out_btd.shape}")
    print(f"维度匹配: {out_btd.shape == x_btd.shape}")
    assert out_btd.shape == x_btd.shape, "[B,T,D]格式输出维度不匹配！"
    print("✓ [B, T, D]格式测试通过")
    
    # 测试5: 不同序列长度
    print("\n【测试5】不同序列长度兼容性")
    print("-" * 60)
    for test_len in [32, 64, 128, 256]:
        x_test = torch.randn(test_len, batch_size, feature_dim)
        with torch.no_grad():
            out_test = adaptive_model(x_test)
        print(f"序列长度={test_len:3d}: 输入{x_test.shape} -> 输出{out_test.shape} ✓")
        assert out_test.shape == x_test.shape
    print("✓ 不同序列长度测试通过")
    
    # 测试6: 梯度反向传播
    print("\n【测试6】梯度反向传播")
    print("-" * 60)
    adaptive_model.train()
    x_grad = torch.randn(32, 2, feature_dim, requires_grad=True)
    out_grad = adaptive_model(x_grad)
    loss = out_grad.sum()
    loss.backward()
    
    print(f"输入梯度存在: {x_grad.grad is not None}")
    print(f"输入梯度形状: {x_grad.grad.shape if x_grad.grad is not None else 'None'}")
    assert x_grad.grad is not None, "梯度未正确传播！"
    print("✓ 梯度反向传播测试通过")
    
    # 测试7: 参数统计
    print("\n【测试7】模型参数统计")
    print("-" * 60)
    adaptive_params = sum(p.numel() for p in adaptive_model.parameters())
    hierarchical_params = sum(p.numel() for p in hierarchical_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    
    print(f"AdaptiveConvGCN参数量: {adaptive_params:,}")
    print(f"HierarchicalTransformer参数量: {hierarchical_params:,}")
    print(f"TransformerEncoder参数量: {transformer_params:,}")
    print(f"相对大小: {adaptive_params / hierarchical_params:.2f}x (vs Hierarchical)")
    
    # 测试8: 特征变化率计算
    print("\n【测试8】特征变化率计算")
    print("-" * 60)
    from utils.adaptive_temporal import compute_change_rate
    
    x_change = torch.randn(batch_size, seq_len, feature_dim)
    delta = compute_change_rate(x_change)
    
    print(f"输入形状: {x_change.shape}")
    print(f"变化率形状: {delta.shape}")
    print(f"变化率范围: [{delta.min().item():.4f}, {delta.max().item():.4f}]")
    print(f"变化率均值: {delta.mean().item():.4f}")
    assert delta.shape == (batch_size, seq_len), "变化率维度不正确！"
    assert delta.min() >= 0.0 and delta.max() <= 2.0, "变化率范围异常！"
    print("✓ 特征变化率计算测试通过")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！模块可以安全集成到项目中。")
    print("=" * 60)

if __name__ == "__main__":
    test_dimension_compatibility()

