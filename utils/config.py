import argparse

parser = argparse.ArgumentParser()
# 你好
parser.add_argument("--dataset", type=str, default="ucf", choices=["ucf", "xd"])
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Local learning rate")
parser.add_argument("--global_rounds", type=int, default=1, help="Global federated learning rounds")
parser.add_argument("--local_epochs", type=int, default=1, help="local training epochs for each client")
parser.add_argument("--algorithm", type=str, default="FedAvg", choices=["FedAvg", "FedProx", "Scaffold"])
parser.add_argument("--clients_num", type=int, default=13, help="Number of client per round when random")
parser.add_argument("--split_mode", type=str, default="event", choices=["random", "event", "scene"])
parser.add_argument('--scheduler_rate', default=0.1)
parser.add_argument('--scheduler_milestones', default=[])
parser.add_argument('--load_model', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--embed_dim', default=512, type=int, help="CLIP embedding size")
parser.add_argument('--visual_length', default=256,type=int, help="length of video")
parser.add_argument('--visual_width', default=512, type=int)
parser.add_argument('--visual_head', default=1, type=int)
parser.add_argument('--visual_layers', default=2, type=int, help="number of visual layers, default 2 for ucf")
# 没用到下面的参数
parser.add_argument('--attn_window', default=8, type=int, help="size of attention window, default 8 for ucf")
parser.add_argument('--prompt_prefix', default=10, type=int)
parser.add_argument('--prompt_postfix', default=10, type=int)


# Hierarchical Transformer parameters (1,长短期时间建模)
parser.add_argument('--local_layers', default=2, type=int, help="number of local attention layers")
parser.add_argument('--global_layers', default=2, type=int, help="number of global attention layers")
parser.add_argument('--window_size', default=16, type=int, help="size of local attention window, 16 for ucf and part of xd")
parser.add_argument('--transformer_dropout', default=0.1, type=float, help="dropout rate for transformer")
parser.add_argument('--temporal_type', type=str, default='hierarchical', choices=['hierarchical', 'standard'],
                   help='Type of temporal modeling: hierarchical (HierarchicalTransformer) or standard (TransformerEncoder)')
