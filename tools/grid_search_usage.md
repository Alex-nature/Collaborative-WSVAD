# 网格搜索参数优化工具

本工具用于自动化进行联邦学习实验的参数优化，支持多个数据集和不同的数据划分方式。

## 功能特点

- 支持UCF和XD两个数据集
- 支持事件、场景和随机三种划分方式
- 自动化参数搜索和实验执行
- 实验结果自动保存和记录
- 支持断点续跑
- 支持选择性运行特定数据集或划分方式

## 参数搜索空间

### 数据集配置

#### XD数据集
- 事件划分: 6个客户端，batch_size=128
- 随机划分: 10个客户端，batch_size=128
- 场景划分: 13个客户端，batch_size=128

#### UCF数据集
- 事件划分: 13个客户端，batch_size=64
- 随机划分: 10个客户端，batch_size=64
- 场景划分: 9个客户端，batch_size=64

### 超参数搜索范围

| 参数 | 搜索范围 | 说明 |
|------|----------|------|
| learning_rate | [5e-6, 1e-5, 5e-5, 1e-4] | 学习率 |
| global_rounds | [15, 20, 25] | 全局训练轮次 |
| local_epochs | [8, 10, 12] | 本地训练轮次 |
| scheduler_rate | [0.1, 0.2] | 学习率衰减率 |

### 学习率调度设置

| 全局轮次 | 学习率衰减点 |
|----------|--------------|
| 15轮 | [7, 12] |
| 20轮 | [10, 15] |
| 25轮 | [12, 20] |

## 使用方法

### 基本用法

```bash
# 运行所有实验
python tools/grid_search.py

# 运行指定范围的实验
python tools/grid_search.py --start 0 --end 10
```

### 选择性运行

```bash
# 只运行UCF数据集的实验
python tools/grid_search.py --dataset ucf

# 只运行场景划分的实验
python tools/grid_search.py --split scene

# 只运行XD数据集的随机划分实验
python tools/grid_search.py --dataset xd --split random
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --start | int | 0 | 起始实验ID |
| --end | int | None | 结束实验ID |
| --dataset | str | 'all' | 指定数据集 (ucf/xd/all) |
| --split | str | 'all' | 指定划分方式 (event/scene/random/all) |

## 输出目录结构

```
experiments/
└── 20250928_153000/           # 时间戳命名的实验目录
    ├── exp_0_config.json      # 实验0的配置和结果
    ├── exp_1_config.json      # 实验1的配置和结果
    └── ...
```

### 配置文件格式

```json
{
    "id": 0,
    "parameters": {
        "dataset": "xd",
        "split_mode": "event",
        "clients_num": 6,
        "batch_size": 128,
        "learning_rate": 5e-6,
        "global_rounds": 15,
        "local_epochs": 8,
        "scheduler_rate": 0.1,
        "scheduler_milestones": [7, 12]
    },
    "timestamp": "2025-09-28 15:30:00",
    "status": "success"
}
```

## 实验数量

- 每个数据集配置：24种参数组合
  - 4(learning_rate) × 3(global_rounds) × 3(local_epochs) × 2(scheduler_rate) = 24
- 总实验数：144个实验
  - 6(数据集配置) × 24(参数组合) = 144

## 注意事项

1. 实验结果保存
   - 每个实验的模型文件保存在save目录
   - 实验配置和状态保存在experiments目录
   - 实验日志实时输出到控制台

2. 运行建议
   - 长时间运行建议使用screen或nohup
   - 可以分批次运行不同的数据集或划分方式
   - 建议先运行少量实验测试配置

3. 结果分析
   - 实验结果需要手动整理和分析
   - 可以通过实验配置文件追踪每个实验的参数和性能
   - 建议使用表格或图表可视化比较不同参数组合的效果

## 实验监控

1. 实时输出
   ```
   === 运行实验 0 ===
   数据集: xd, 划分方式: event
   参数配置:
   learning_rate: 5e-6
   global_rounds: 15
   local_epochs: 8
   scheduler_rate: 0.1
   scheduler_milestones: [7,12]
   batch_size: 128
   clients_num: 6
   ```

2. 进度跟踪
   - 通过实验ID和总数了解整体进度
   - 通过配置文件检查每个实验的状态
   - 可以随时中断和继续实验 