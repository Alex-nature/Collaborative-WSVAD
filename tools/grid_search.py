import os
import itertools
import json
from datetime import datetime
import subprocess
import argparse

class GridSearch:
    def __init__(self):
        # 定义数据集配置
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

        # 优化后的参数搜索空间
        self.param_grid = {
            'learning_rate': [1e-5, 1e-4],  # 论文值和一个探索值
            'global_rounds': [15, 20],      # 论文值和代码默认值
            'local_epochs': [10],           # 论文和代码共同使用的值
            'scheduler_rate': [0.1, 0.2],   # 保持两个值以探索学习率衰减的影响
            'scheduler_milestones': [
                [7, 12],    # 用于15轮
                [10, 15],   # 用于20轮
            ]  # 根据global_rounds设置，大约在一半和3/4处降低学习率
        }
        
        # 创建实验记录目录
        self.exp_dir = os.path.join('experiments', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.exp_dir, exist_ok=True)

        # 创建实验说明文件
        self._create_experiment_info()

    def _create_experiment_info(self):
        """创建实验说明文件，记录参数配置信息"""
        info_path = os.path.join(self.exp_dir, 'experiment_info.md')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("# 网格搜索实验配置\n\n")
            
            f.write("## 参数搜索空间\n\n")
            for param, values in self.param_grid.items():
                f.write(f"### {param}\n")
                f.write(f"- 值: {values}\n")
                if param == 'learning_rate':
                    f.write("- 说明: 1e-5(论文值)和5e-5(探索值)\n")
                elif param == 'global_rounds':
                    f.write("- 说明: 15(论文值)和20(代码默认值)\n")
                elif param == 'local_epochs':
                    f.write("- 说明: 10(论文和代码共同值)\n")
                f.write("\n")
            
            f.write("## 数据集配置\n\n")
            for config in self.dataset_configs:
                f.write(f"### {config['dataset']}-{config['split_mode']}\n")
                f.write(f"- clients_num: {config['clients_num']}\n")
                f.write(f"- batch_size: {config['batch_size']}\n\n")

    def generate_combinations(self):
        """生成参数组合"""
        valid_combinations = []
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # 生成所有参数组合
        for dataset_config in self.dataset_configs:
            for combination in itertools.product(*param_values):
                params = dict(zip(param_keys, combination))
                # 根据global_rounds选择合适的scheduler_milestones
                rounds = params['global_rounds']
                if rounds == 15 and params['scheduler_milestones'] == [7, 12]:
                    # 合并数据集配置和参数配置
                    full_params = {**dataset_config, **params}
                    valid_combinations.append(full_params)
                elif rounds == 20 and params['scheduler_milestones'] == [10, 15]:
                    full_params = {**dataset_config, **params}
                    valid_combinations.append(full_params)
        
        return valid_combinations

    def run_experiment(self, params, exp_id):
        """运行单个实验"""
        # 构建命令行参数
        cmd = ['python', 'train.py']
        for key, value in params.items():
            if isinstance(value, list):
                value = str(value).replace(' ', '')
            cmd.extend([f'--{key}', str(value)])
        
        # 记录实验配置
        exp_config = {
            'id': exp_id,
            'parameters': params,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存实验配置
        config_path = os.path.join(self.exp_dir, f'exp_{exp_id}_config.json')
        with open(config_path, 'w') as f:
            json.dump(exp_config, f, indent=4)
        
        print(f"\n=== 运行实验 {exp_id} ===")
        print(f"数据集: {params['dataset']}, 划分方式: {params['split_mode']}")
        print("参数配置:")
        for key, value in params.items():
            print(f"{key}: {value}")
        
        # 运行训练脚本
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 实时输出日志
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # 获取返回码
            return_code = process.poll()
            
            # 记录实验结果
            exp_config['status'] = 'success' if return_code == 0 else 'failed'
            with open(config_path, 'w') as f:
                json.dump(exp_config, f, indent=4)
            
        except Exception as e:
            print(f"实验运行出错: {str(e)}")
            exp_config['status'] = 'error'
            exp_config['error'] = str(e)
            with open(config_path, 'w') as f:
                json.dump(exp_config, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='网格搜索参数优化工具')
    parser.add_argument('--start', type=int, default=0, help='起始实验ID')
    parser.add_argument('--end', type=int, default=None, help='结束实验ID')
    parser.add_argument('--dataset', type=str, choices=['ucf', 'xd', 'all'], default='all',
                      help='指定要运行的数据集 (ucf, xd, 或 all)')
    parser.add_argument('--split', type=str, choices=['event', 'scene', 'random', 'all'], default='all',
                      help='指定要运行的划分方式 (event, scene, random, 或 all)')
    args = parser.parse_args()

    grid_search = GridSearch()
    all_combinations = grid_search.generate_combinations()
    
    # 根据命令行参数过滤实验组合
    filtered_combinations = []
    for combo in all_combinations:
        if args.dataset != 'all' and combo['dataset'] != args.dataset:
            continue
        if args.split != 'all' and combo['split_mode'] != args.split:
            continue
        filtered_combinations.append(combo)
    
    # 计算实验范围
    total_experiments = len(filtered_combinations)
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_experiments
    
    print(f"\n=== 网格搜索配置 ===")
    print(f"数据集: {args.dataset}")
    print(f"划分方式: {args.split}")
    print(f"总共生成了 {total_experiments} 种参数组合")
    print(f"将运行实验 {start_idx} 到 {end_idx-1}")
    print("="*50)
    
    # 运行指定范围的实验
    for i in range(start_idx, end_idx):
        if i >= total_experiments:
            break
        grid_search.run_experiment(filtered_combinations[i], i)

if __name__ == "__main__":
    main() 