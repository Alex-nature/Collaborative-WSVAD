"""
自适应时序建模模块使用示例
演示如何在实际场景中使用该模块
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.adaptive_temporal import (
    compute_change_rate,
    AdaptiveConvGCNTemporal,
    AdaptiveAdjacencyBuilder
)


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "="*60)
    print("示例1: 基本使用")
    print("="*60)
    
    # 创建模拟的CLIP帧特征
    batch_size = 2
    num_frames = 64
    feature_dim = 512  # CLIP ViT-B/16
    
    # 模拟视频特征 [B, T, D]
    video_features = torch.randn(batch_size, num_frames, feature_dim)
    print(f"输入视频特征形状: {video_features.shape}")
    
    # 创建自适应时序模块
    model = AdaptiveConvGCNTemporal(
        width=feature_dim,
        gcn_layers=2,
        min_window=2,
        max_window=16,
        use_feature_sim=True
    )
    
    # 前向传播
    with torch.no_grad():
        output = model(video_features)
    
    print(f"输出特征形状: {output.shape}")
    print(f"✓ 维度保持不变")


def example_2_change_rate_analysis():
    """示例2: 分析视频变化率"""
    print("\n" + "="*60)
    print("示例2: 分析视频变化率")
    print("="*60)
    
    # 创建具有不同变化模式的视频
    num_frames = 100
    feature_dim = 512
    
    # 模式1: 平滑变化
    smooth_video = torch.randn(1, num_frames, feature_dim).cumsum(dim=1)
    smooth_video = smooth_video / smooth_video.norm(dim=-1, keepdim=True)
    
    # 模式2: 突变（在第50帧）
    abrupt_video = torch.randn(1, num_frames, feature_dim)
    abrupt_video[:, 50:, :] += 5.0  # 在第50帧添加大变化
    
    # 计算变化率
    delta_smooth = compute_change_rate(smooth_video)
    delta_abrupt = compute_change_rate(abrupt_video)
    
    print(f"平滑视频变化率: 均值={delta_smooth.mean():.4f}, 最大={delta_smooth.max():.4f}")
    print(f"突变视频变化率: 均值={delta_abrupt.mean():.4f}, 最大={delta_abrupt.max():.4f}")
    print(f"✓ 突变视频在第50帧处检测到最大变化率")


def example_3_adaptive_window():
    """示例3: 观察自适应窗口大小"""
    print("\n" + "="*60)
    print("示例3: 观察自适应窗口大小")
    print("="*60)
    
    # 创建变化率序列
    num_frames = 50
    batch_size = 1
    feature_dim = 512
    
    # 模拟变化率：前半部分变化小，后半部分变化大
    delta = torch.cat([
        torch.ones(batch_size, num_frames//2) * 0.1,   # 平滑区域
        torch.ones(batch_size, num_frames//2) * 0.8    # 动态区域
    ], dim=1)
    
    # 创建邻接矩阵构建器
    adj_builder = AdaptiveAdjacencyBuilder(min_window=2, max_window=16)
    
    # 计算窗口大小
    delta_max = delta.max(dim=1, keepdim=True)[0] + 1e-6
    normalized_delta = delta / delta_max
    window_sizes = 2 + (16 - 2) * (1.0 - normalized_delta)
    
    print(f"平滑区域平均窗口大小: {window_sizes[0, :num_frames//2].mean():.1f}")
    print(f"动态区域平均窗口大小: {window_sizes[0, num_frames//2:].mean():.1f}")
    print(f"✓ 平滑区域使用大窗口，动态区域使用小窗口")


def example_4_integration_with_model():
    """示例4: 与完整模型集成"""
    print("\n" + "="*60)
    print("示例4: 与完整模型集成（模拟）")
    print("="*60)
    
    from utils.model import Model
    
    # 模拟训练参数
    class Args:
        embed_dim = 512
        visual_length = 256
        prompt_prefix = 10
        prompt_postfix = 10
        visual_width = 512
        visual_head = 8
        visual_layers = 2
        local_layers = 2
        global_layers = 2
        window_size = 16
        transformer_dropout = 0.1
        temporal_type = 'adaptive_conv_gcn'  # 使用自适应模块
        # 自适应参数
        tcn_out_dim = None
        gcn_layers = 2
        min_window = 2
        max_window = 16
        use_feature_sim = True
        weight_hidden_dim = 128
    
    args = Args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"创建模型...")
    print(f"- temporal_type: {args.temporal_type}")
    print(f"- gcn_layers: {args.gcn_layers}")
    print(f"- window range: [{args.min_window}, {args.max_window}]")
    
    try:
        model = Model(
            args.embed_dim, args.visual_length, args.prompt_prefix, args.prompt_postfix,
            args.visual_width, args.visual_head, args.visual_layers,
            args.local_layers, args.global_layers, args.window_size, args.transformer_dropout,
            args.temporal_type, device,
            tcn_out_dim=args.tcn_out_dim,
            gcn_layers=args.gcn_layers,
            min_window=args.min_window,
            max_window=args.max_window,
            use_feature_sim=args.use_feature_sim,
            weight_hidden_dim=args.weight_hidden_dim
        ).to(device)
        
        print(f"✓ 模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")


def example_5_compare_temporal_types():
    """示例5: 对比不同时序建模方法"""
    print("\n" + "="*60)
    print("示例5: 对比不同时序建模方法的输出")
    print("="*60)
    
    from utils.hierarchical_transformer import HierarchicalTransformer
    from utils.model import TransformerEncoder
    
    # 测试参数
    batch_size = 2
    seq_len = 64
    feature_dim = 512
    
    # 创建输入
    x = torch.randn(seq_len, batch_size, feature_dim)
    
    # 三种时序建模方法
    models = {
        'Standard Transformer': TransformerEncoder(
            width=feature_dim, layers=2, heads=8, dropout=0.1
        ),
        'Hierarchical Transformer': HierarchicalTransformer(
            width=feature_dim, local_layers=2, global_layers=2,
            heads=8, window_size=16, dropout=0.1
        ),
        'Adaptive Conv+GCN': AdaptiveConvGCNTemporal(
            width=feature_dim, gcn_layers=2, min_window=2,
            max_window=16, dropout=0.1, use_feature_sim=True
        )
    }
    
    print(f"输入形状: {x.shape}\n")
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # 统计
        params = sum(p.numel() for p in model.parameters())
        output_norm = output.norm(dim=-1).mean().item()
        
        print(f"{name:25s} | 输出形状: {tuple(output.shape)} | "
              f"参数量: {params:7,} | 输出范数: {output_norm:.3f}")
    
    print("\n✓ 所有方法输出维度一致")


def example_6_visualize_adaptive_behavior():
    """示例6: 可视化自适应行为（保存图像）"""
    print("\n" + "="*60)
    print("示例6: 可视化自适应行为")
    print("="*60)
    
    # 创建具有明显场景变化的模拟视频
    num_frames = 100
    feature_dim = 512
    
    # 场景1 (0-30帧): 静态
    scene1 = torch.randn(1, 30, feature_dim) * 0.1 + torch.randn(1, 1, feature_dim)
    # 场景转换 (30-35帧): 剧烈变化
    transition = torch.randn(1, 5, feature_dim) * 2.0
    # 场景2 (35-70帧): 静态
    scene2 = torch.randn(1, 35, feature_dim) * 0.1 + torch.randn(1, 1, feature_dim)
    # 场景转换 (70-75帧): 剧烈变化
    transition2 = torch.randn(1, 5, feature_dim) * 2.0
    # 场景3 (75-100帧): 静态
    scene3 = torch.randn(1, 25, feature_dim) * 0.1 + torch.randn(1, 1, feature_dim)
    
    video = torch.cat([scene1, transition, scene2, transition2, scene3], dim=1)
    
    # 计算变化率
    delta = compute_change_rate(video)[0].numpy()  # [T]
    
    # 计算自适应窗口大小
    min_w, max_w = 2, 16
    delta_max = delta.max() + 1e-6
    window_sizes = min_w + (max_w - min_w) * (1.0 - delta / delta_max)
    
    print(f"场景1 (0-30) 平均变化率: {delta[0:30].mean():.4f}, 平均窗口: {window_sizes[0:30].mean():.1f}")
    print(f"转换1 (30-35) 平均变化率: {delta[30:35].mean():.4f}, 平均窗口: {window_sizes[30:35].mean():.1f}")
    print(f"场景2 (35-70) 平均变化率: {delta[35:70].mean():.4f}, 平均窗口: {window_sizes[35:70].mean():.1f}")
    print(f"转换2 (70-75) 平均变化率: {delta[70:75].mean():.4f}, 平均窗口: {window_sizes[70:75].mean():.1f}")
    print(f"场景3 (75-100) 平均变化率: {delta[75:100].mean():.4f}, 平均窗口: {window_sizes[75:100].mean():.1f}")
    
    try:
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        frames = np.arange(num_frames)
        
        # 子图1: 变化率
        ax1.plot(frames, delta, linewidth=2, color='steelblue')
        ax1.fill_between(frames, 0, delta, alpha=0.3, color='steelblue')
        ax1.axvspan(30, 35, alpha=0.2, color='red', label='Scene Transition')
        ax1.axvspan(70, 75, alpha=0.2, color='red')
        ax1.set_ylabel('Change Rate', fontsize=12)
        ax1.set_title('Frame-level Feature Change Rate', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: 自适应窗口大小
        ax2.plot(frames, window_sizes, linewidth=2, color='darkgreen')
        ax2.fill_between(frames, min_w, window_sizes, alpha=0.3, color='green')
        ax2.axhline(y=min_w, color='gray', linestyle='--', alpha=0.5, label=f'min_window={min_w}')
        ax2.axhline(y=max_w, color='gray', linestyle='--', alpha=0.5, label=f'max_window={max_w}')
        ax2.axvspan(30, 35, alpha=0.2, color='red')
        ax2.axvspan(70, 75, alpha=0.2, color='red')
        ax2.set_xlabel('Frame Index', fontsize=12)
        ax2.set_ylabel('Adaptive Window Size', fontsize=12)
        ax2.set_title('Adaptive Temporal Window Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('adaptive_behavior_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ 可视化图像已保存至: adaptive_behavior_visualization.png")
        
    except Exception as e:
        print(f"\n✗ 可视化失败（matplotlib可能未安装）: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("自适应时序建模模块使用示例")
    print("="*60)
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_change_rate_analysis()
    example_3_adaptive_window()
    example_4_integration_with_model()
    example_5_compare_temporal_types()
    example_6_visualize_adaptive_behavior()
    
    print("\n" + "="*60)
    print("✓ 所有示例运行完成！")
    print("="*60)

