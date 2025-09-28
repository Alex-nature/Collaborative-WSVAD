# 贝叶斯优化参数搜索工具

本工具使用贝叶斯优化方法来自动化寻找最优的模型参数配置。每次试验（trial）会在所有数据集配置上运行，以找到整体性能最优的参数组合。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 功能特点

- 使用贝叶斯优化进行智能参数搜索
- 支持连续参数和离散参数混合优化
- 自动生成优化过程可视化
- 支持断点续训
- 详细的实验记录和结果分析
- 每个trial自动运行所有数据集配置

## 数据集配置

每个trial会依次运行以下配置：

1. **XD数据集**：
   - Event划分: 6个客户端，batch_size=128
   - Random划分: 10个客户端，batch_size=128
   - Scene划分: 13个客户端，batch_size=128

2. **UCF数据集**：
   - Event划分: 13个客户端，batch_size=64
   - Random划分: 10个客户端，batch_size=64
   - Scene划分: 9个客户端，batch_size=64

## 参数搜索空间

### 训练相关参数
- learning_rate: [1e-5, 1e-4] (对数尺度)
- scheduler_rate: [0.1, 0.2] (线性尺度)
- global_rounds: [15, 20]
- local_epochs: [8, 10, 12]

### Transformer相关参数
- local_layers: [1, 2, 3, 4] (局部注意力层数)
- global_layers: [1, 2, 3, 4] (全局注意力层数)
- window_size: [4, 8, 16, 32, 64, 128] (局部注意力窗口大小)
- transformer_dropout: [0.1, 0.15] (Transformer dropout率)
- visual_head: [1, 2, 4, 8] (注意力头数)

### Milestone映射规则
- 15轮: [7, 12]
- 20轮: [10, 15]

## 优化目标

工具会计算每个参数组合在所有配置上的平均性能作为优化目标：
- XD数据集使用AP指标
- UCF数据集使用ROC指标
- 最终目标是最大化所有配置的平均性能

## 使用方法

### 基本用法

```bash
# 运行贝叶斯优化（自动运行所有配置）
python tools/bayesian_optimization.py --trials 10
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --trials | int | 20 | 优化试验次数 |

## 输出目录结构

```
experiments/
└── bayesian_opt_20250928_153000/           # 优化实验目录
    ├── experiment_info.md                   # 实验配置说明
    ├── optimization.db                      # 优化历史数据库
    ├── trial_0/                            # 第0次试验
    │   ├── config.json                     # 试验参数配置
    │   ├── xd_event/                       # XD-event配置结果
    │   │   ├── output.log                  # 训练日志
    │   │   └── result.json                 # 性能结果
    │   ├── xd_random/                      # XD-random配置结果
    │   │   ├── output.log
    │   │   └── result.json
    │   ├── xd_scene/                       # XD-scene配置结果
    │   ├── ucf_event/                      # UCF-event配置结果
    │   ├── ucf_random/                     # UCF-random配置结果
    │   └── ucf_scene/                      # UCF-scene配置结果
    └── optimization_results/               # 优化结果分析
        ├── best_params.json                # 最佳参数
        ├── optimization_history.html       # 优化历史可视化
        └── param_importances.html          # 参数重要性分析
```

## 结果分析

### 性能评估
- 每个trial的性能是6个配置性能的平均值
- 可以通过result.json查看每个配置的具体性能
- optimization_history.html显示平均性能的优化过程

### 参数分析
- param_importances.html展示参数对整体性能的影响
- best_params.json记录最佳参数组合
- 可以比较不同配置对参数的敏感度

## 运行策略建议

1. **计算资源评估**：
   - 每个trial需要运行6个完整配置
   - 总运行时间 ≈ 单个配置时间 × 6 × trials数量
   - 建议先用较少的trials测试

2. **优化过程监控**：
   - 关注每个配置的单独性能
   - 观察参数对不同配置的影响
   - 记录资源使用情况

3. **结果分析**：
   - 检查参数是否对所有配置都有效
   - 分析性能瓶颈
   - 考虑配置特定的优化策略

## 注意事项

1. **运行时间**：
   - 每个trial运行时间较长（需要完成6个配置）
   - 建议使用screen或nohup运行
   - 支持断点续训

2. **存储空间**：
   - 每个trial会产生6个配置的日志和结果
   - 定期清理不需要的实验数据
   - 保留重要的实验记录

3. **结果解释**：
   - 最佳参数是对所有配置的平均最优
   - 可能需要针对特定配置进行微调
   - 考虑参数在不同配置间的迁移性

4. **优化建议**：
   - 从较少的trials开始测试
   - 根据初步结果调整参数范围
   - 保存重要的中间结果 