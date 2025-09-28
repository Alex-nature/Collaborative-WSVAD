import os
import json
import time
from datetime import datetime
import subprocess
import optuna
import joblib
import plotly.graph_objects as go
import numpy as np
from optuna.visualization import plot_optimization_history, plot_param_importances

class BayesianOptimizer:
    def __init__(self):
        # 数据集配置
        self.dataset_configs = [
            # XD数据集配置
            {'dataset': 'xd', 'split_mode': 'event', 'clients_num': 6, 'batch_size': 128},
            {'dataset': 'xd', 'split_mode': 'random', 'clients_num': 10, 'batch_size': 128},
            {'dataset': 'xd', 'split_mode': 'scene', 'clients_num': 13, 'batch_size': 128},
            # UCF数据集配置
            {'dataset': 'ucf', 'split_mode': 'event', 'clients_num': 13, 'batch_size': 64},
            {'dataset': 'ucf', 'split_mode': 'random', 'clients_num': 10, 'batch_size': 64},
            {'dataset': 'ucf', 'split_mode': 'scene', 'clients_num': 9, 'batch_size': 64},
        ]

        # milestone映射规则
        self.milestone_mapping = {
            15: [7, 12],
            20: [10, 15]
        }

        # 创建实验目录
        self.exp_dir = os.path.join('experiments', 'bayesian_opt_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 创建实验说明文件
        self._create_experiment_info()

    def _create_experiment_info(self):
        """创建实验说明文件"""
        info_path = os.path.join(self.exp_dir, 'experiment_info.md')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("# 贝叶斯优化实验配置\n\n")
            
            f.write("## 参数空间\n\n")
            f.write("### 训练相关参数\n")
            f.write("- learning_rate: [1e-5, 1e-4] (对数尺度)\n")
            f.write("- scheduler_rate: [0.1, 0.2] (线性尺度)\n")
            f.write("- global_rounds: [15, 20]\n")
            f.write("- local_epochs: [8, 10, 12]\n\n")
            
            f.write("### Transformer相关参数\n")
            f.write("- local_layers: [1, 2, 3, 4] (局部注意力层数)\n")
            f.write("- global_layers: [1, 2, 3, 4] (全局注意力层数)\n")
            f.write("- window_size: [4, 8, 16, 32, 64, 128] (局部注意力窗口大小)\n")
            f.write("- transformer_dropout: [0.1, 0.15] (Transformer dropout率)\n")
            f.write("- visual_head: [1, 2, 4, 8] (注意力头数)\n\n")
            
            f.write("### Milestone映射规则\n")
            for rounds, milestones in self.milestone_mapping.items():
                f.write(f"- {rounds}轮: {milestones}\n")
            f.write("\n")
            
            f.write("## 数据集配置\n\n")
            for config in self.dataset_configs:
                f.write(f"### {config['dataset']}-{config['split_mode']}\n")
                f.write(f"- clients_num: {config['clients_num']}\n")
                f.write(f"- batch_size: {config['batch_size']}\n\n")

    def objective(self, trial, dataset_config):
        """优化目标函数"""
        # 采样超参数
        params = {
            # 学习率：论文使用1e-5，代码默认1e-4，在这个范围内对数均匀采样
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
            
            # 全局轮次：论文使用15轮，代码默认20轮
            'global_rounds': trial.suggest_categorical('global_rounds', [15, 20]),
            
            # 本地轮次：论文和代码都使用10轮，适当调整范围
            'local_epochs': trial.suggest_categorical('local_epochs', [8, 10, 12]),
            
            # 学习率衰减率：默认0.1，适当扩大范围以探索更温和的衰减
            'scheduler_rate': trial.suggest_float('scheduler_rate', 0.1, 0.2),
            
            # Transformer相关参数
            # local_layers：代码默认2层，扩展搜索范围以探索更深的局部特征提取
            'local_layers': trial.suggest_categorical('local_layers', [1, 2, 3, 4]),
            
            # global_layers：代码默认2层，扩展范围以探索不同程度的全局建模能力
            'global_layers': trial.suggest_categorical('global_layers', [1, 2, 3, 4]),
            
            # window_size：代码默认16，添加更多选项以适应不同尺度的时序依赖
            'window_size': trial.suggest_categorical('window_size', [4, 8, 16, 32, 64, 128]),
            
            # transformer_dropout：代码默认0.1，适当调整以控制过拟合
            'transformer_dropout': trial.suggest_float('transformer_dropout', 0.1, 0.15),
            
            # visual_head：注意力头数，代码默认1，探索多头注意力的效果
            'visual_head': trial.suggest_categorical('visual_head', [1, 2, 4, 8])
        }
        
        # 根据global_rounds设置milestone
        params['scheduler_milestones'] = self.milestone_mapping[params['global_rounds']]
        
        # 创建trial目录
        trial_dir = os.path.join(self.exp_dir, f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)
        
        # 记录参数配置
        config_path = os.path.join(trial_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'trial_id': trial.number,
                'parameters': params,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)

        # 运行所有数据集配置
        all_metrics = []
        for config in self.dataset_configs:
            # 合并数据集配置和参数
            full_params = {**config, **params}
            
            # 构建命令行参数
            cmd = ['python', 'train.py']
            for key, value in full_params.items():
                if isinstance(value, list):
                    value = str(value).replace(' ', '')
                cmd.extend([f'--{key}', str(value)])
            
            # 创建配置特定的目录
            config_dir = os.path.join(trial_dir, f"{config['dataset']}_{config['split_mode']}")
            os.makedirs(config_dir, exist_ok=True)
            
            try:
                # 运行训练脚本
                start_time = time.time()
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1  # 行缓冲
                )
                
                # 记录输出
                log_path = os.path.join(config_dir, 'output.log')
                current_metric = None
                seen_outputs = set()  # 用于跟踪已经看到的输出

                with open(log_path, 'w') as log_file:
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            output = output.strip()
                            # 只记录没见过的输出
                            if output not in seen_outputs:
                                print(output)
                                log_file.write(output + '\n')
                                log_file.flush()
                                seen_outputs.add(output)
                            
                            # 解析性能指标
                            if 'current ROC:' in output:
                                current_metric = float(output.split('current ROC:')[1].strip())
                            elif 'current AP:' in output:
                                current_metric = float(output.split('current AP:')[1].strip())
                
                training_time = time.time() - start_time
                
                # 记录结果
                result = {
                    'metric': current_metric,
                    'training_time': training_time,
                    'status': 'completed'
                }
                all_metrics.append(current_metric if current_metric is not None else float('-inf'))
                
            except Exception as e:
                result = {
                    'error': str(e),
                    'status': 'failed'
                }
                all_metrics.append(float('-inf'))
            
            # 保存结果
            result_path = os.path.join(config_dir, 'result.json')
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
        
        # 返回所有指标的平均值作为优化目标
        valid_metrics = [m for m in all_metrics if m != float('-inf')]
        return sum(valid_metrics) / len(valid_metrics) if valid_metrics else float('-inf')

    def optimize(self, n_trials=20):
        """执行优化过程"""
        storage_path = os.path.join(self.exp_dir, f'optimization.db')
        
        # 创建study对象
        study = optuna.create_study(
            storage=f"sqlite:///{storage_path}",
            direction="maximize",
            load_if_exists=True
        )
        
        # 运行优化
        study.optimize(
            lambda trial: self.objective(trial, None),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # 保存优化结果
        self.save_optimization_results(study)
        
        return study

    def save_optimization_results(self, study, dataset_config):
        """保存优化结果和可视化"""
        study_name = f"{dataset_config['dataset']}_{dataset_config['split_mode']}"
        results_dir = os.path.join(self.exp_dir, study_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存最佳参数
        best_params = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'dataset_config': dataset_config
        }
        
        with open(os.path.join(results_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # 保存优化历史图
        history_fig = plot_optimization_history(study)
        history_fig.write_html(os.path.join(results_dir, 'optimization_history.html'))
        
        # 保存参数重要性图
        importance_fig = plot_param_importances(study)
        importance_fig.write_html(os.path.join(results_dir, 'param_importances.html'))
        
        # 保存study对象
        joblib.dump(study, os.path.join(results_dir, 'study.pkl'))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='贝叶斯优化参数搜索工具')
    parser.add_argument('--dataset', type=str, choices=['ucf', 'xd', 'all'], default='all',
                      help='指定要优化的数据集 (ucf, xd, 或 all)')
    parser.add_argument('--split', type=str, choices=['event', 'scene', 'random', 'all'], default='all',
                      help='指定要优化的划分方式 (event, scene, random, 或 all)')
    parser.add_argument('--trials', type=int, default=20,
                      help='随机配置的总次数（一个配置运行整个6次实验）')
    args = parser.parse_args()
    
    optimizer = BayesianOptimizer()
    
    # 根据命令行参数过滤数据集配置
    configs_to_run = []
    for config in optimizer.dataset_configs:
        if args.dataset != 'all' and config['dataset'] != args.dataset:
            continue
        if args.split != 'all' and config['split_mode'] != args.split:
            continue
        configs_to_run.append(config)
    
    print(f"\n=== 贝叶斯优化配置 ===")
    print(f"数据集: {args.dataset}")
    print(f"划分方式: {args.split}")
    print(f"每个配置的试验次数: {args.trials}")
    print(f"要优化的配置数量: {len(configs_to_run)}")
    print("="*50)
    
    # 运行优化
    for config in configs_to_run:
        print(f"\n开始优化 {config['dataset']}-{config['split_mode']} 配置")
        study = optimizer.optimize(n_trials=args.trials)
        print(f"最佳性能: {study.best_value:.4f}")
        print(f"最佳参数: {study.best_params}")
        print("="*50)

if __name__ == "__main__":
    main() 