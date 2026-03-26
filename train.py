import os
import random
import json
from re import A
import numpy as np
import torch
import utils.config as config
from fed.server import FedAvgServer
from utils.dataset import make_xd_dataloader, make_ucf_dataloader
from utils.model import Model
from datetime import datetime


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(88888888)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = config.parser.parse_args()
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    start_time = datetime.now()
    # 修改时间格式，将冒号替换为横杠（windows版本）
    dir_name = start_time.strftime("%Y-%m-%d-%H-%M-%S")
    dir_name = f"{args.dataset}-{args.split_mode}-{dir_name}"
    path = os.path.join('save', dir_name)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'README.txt'), 'w') as f:
        for key, value in args.__dict__.items():
            print(f'{key}: {value}', file=f)

    train_loaders = []

    if args.dataset == "xd":
        train_loaders, test_loader = make_xd_dataloader(
            args.split_mode, args.clients_num, args.batch_size, args.visual_length)

    else:
        train_loaders, test_loader = make_ucf_dataloader(
            args.split_mode, args.clients_num, args.batch_size, args.visual_length)

    model = Model(
        args.embed_dim, 
        args.visual_length, 
        args.prompt_prefix, 
        args.prompt_postfix, 
        args.visual_width, 
        args.visual_head, 
        args.visual_layers, 
        device,
        # TCA参数
        use_tca=args.use_tca,
        tca_window_size=args.tca_window_size,
        tca_dropout=args.tca_dropout,
        use_distance_adj=args.use_distance_adj,
        tca_gamma=args.tca_gamma,
        tca_bias=args.tca_bias,
        tca_norm=args.tca_norm
    ).to(device)

    # 初始化正负分支PromptLearner分离
    model.initialize_separate_prompt_learners()

    if args.load_model == 1:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)

    if args.algorithm == "FedAvg":
        server = FedAvgServer(args.dataset, train_loaders, test_loader, args.clients_num,
                              args.global_rounds, args.local_epochs, args.learning_rate,
                              args.split_mode, args.scheduler_milestones, args.scheduler_rate,
                              device, model,
                              use_dp=args.use_dp,
                              dp_clip_norm=args.dp_clip_norm,
                              dp_noise_multiplier=args.dp_noise_multiplier,
                              dp_noise_mode=args.dp_noise_mode,
                              dp_delta=args.dp_delta,
                              dp_seed=args.dp_seed,
                              dp_log_norm_stats=args.dp_log_norm_stats,
                              use_adaptive_clip=args.use_adaptive_clip,
                              adaptive_clip_quantile=args.adaptive_clip_quantile,
                              adaptive_clip_beta=args.adaptive_clip_beta,
                              adaptive_clip_warmup_rounds=args.adaptive_clip_warmup_rounds,
                              adaptive_clip_min_norm=args.adaptive_clip_min_norm,
                              adaptive_clip_max_norm=args.adaptive_clip_max_norm)
        best_score = server.train(path)
        
        # 确保best_score是有效的数值
        if best_score is not None:
            try:
                best_score = float(best_score)
            except (ValueError, TypeError):
                print(f"警告：无效的评估指标值: {best_score}，使用模型文件名中的值")
                # 尝试从模型文件名中提取分数
                model_files = [f for f in os.listdir(path) if f.startswith('model_final_') and f.endswith('.pth')]
                if model_files:
                    try:
                        score_str = model_files[0].split('_')[-1].replace('.pth', '')
                        best_score = float(score_str)
                    except (ValueError, IndexError):
                        print("无法从模型文件名中提取分数")
                        best_score = 0.0
        else:
            print("警告：评估指标为None，尝试从模型文件名中提取分数")
            # 尝试从模型文件名中提取分数
            model_files = [f for f in os.listdir(path) if f.startswith('model_final_') and f.endswith('.pth')]
            if model_files:
                try:
                    score_str = model_files[0].split('_')[-1].replace('.pth', '')
                    best_score = float(score_str)
                except (ValueError, IndexError):
                    print("无法从模型文件名中提取分数")
                    best_score = 0.0
            else:
                best_score = 0.0
        
        # 保存最终结果
        result = {
            'best_score': best_score,
            'dataset': args.dataset,
            'split_mode': args.split_mode,
            'use_dp': args.use_dp,
            'dp_clip_norm': args.dp_clip_norm,
            'dp_noise_multiplier': args.dp_noise_multiplier,
            'dp_delta': args.dp_delta,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(path, 'final_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
            
        print(f"最终评估指标已保存: {best_score}")
