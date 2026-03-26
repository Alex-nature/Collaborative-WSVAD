from collections import OrderedDict

import torch
import numpy as np
from fed.client import FedAvgClient
from inference import inference


class BaseServer:
    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 model,
                 ):
        super().__init__()
        self.clients = []
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.global_rounds = global_rounds
        self.model = model
        self.local_epochs = local_epochs
        self.clients_num = clients_num
        self.local_weights = []
        self.local_losses = []
        self.len_dataset = []


class FedAvgServer(BaseServer):
    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 learning_rate: float,
                 split_mode,
                 scheduler_milestones,
                 scheduler_rate,
                 device,
                 model,
                 use_dp: bool = False,
                 dp_clip_norm: float = 1.0,
                 dp_noise_multiplier: float = 0.0,
                 dp_noise_mode: str = "local",
                 dp_delta: float = 1e-5,
                 dp_seed: int = 20260326,
                 dp_log_norm_stats: bool = False
                 ):
        super().__init__(dataset, train_loaders, test_loader,
                         clients_num, global_rounds, local_epochs, model)
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.split_mode = split_mode
        self.model = model
        self.global_parameter = None
        self.best = 0
        self.best_model = None
        self.device = device
        self.use_dp = use_dp
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_noise_mode = dp_noise_mode
        self.dp_delta = dp_delta
        self.dp_seed = dp_seed
        self.dp_log_norm_stats = dp_log_norm_stats
        self.dp_generator = torch.Generator(device='cpu')
        self.dp_generator.manual_seed(self.dp_seed)

        if self.dataset == 'ucf':
            label_map = dict(
                {'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})
        else:
            label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
                              'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

        for i in range(clients_num):

            client = FedAvgClient(model, learning_rate, train_loaders[i], dataset,
                                  local_epochs, label_map, scheduler_milestones,
                                  scheduler_rate, device,
                                  use_dp=use_dp,
                                  dp_clip_norm=dp_clip_norm,
                                  dp_noise_multiplier=dp_noise_multiplier,
                                  dp_noise_mode=dp_noise_mode,
                                  dp_seed=dp_seed)

            self.clients.append(client)

        self.global_parameter = self.get_trainable_parameters()
        self.send_global_parameter(self.global_parameter)

    @staticmethod
    def clone_parameter_dict(parameter_dict):
        cloned = OrderedDict()
        for name, value in parameter_dict.items():
            cloned[name] = value.data.clone()
        return cloned

    @staticmethod
    def zero_like_parameter_dict(parameter_dict):
        zeros = OrderedDict()
        for name, value in parameter_dict.items():
            zeros[name] = torch.zeros_like(value)
        return zeros

    @staticmethod
    def add_parameter_dict(base_dict, delta_dict):
        output = OrderedDict()
        for name in base_dict.keys():
            output[name] = base_dict[name].data.clone() + delta_dict[name].data.clone()
        return output

    def get_trainable_parameters(self):
        trainable_parameters = OrderedDict()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                trainable_parameters[name] = p.data.clone()
        return trainable_parameters

    def aggregate_parameters(self):
        temp_dict = OrderedDict()
        total_num = sum(self.len_dataset)
        for key, value in self.local_weights[0].items():
            temp_dict[key] = torch.zeros_like(value)

        for i in range(len(self.local_weights)):
            for key, value in self.local_weights[i].items():
                temp_dict[key] += value * self.len_dataset[i] / total_num

        return temp_dict

    def aggregate_updates(self):
        total_num = sum(self.len_dataset)
        temp_dict = self.zero_like_parameter_dict(self.local_weights[0])

        for i in range(len(self.local_weights)):
            for key, value in self.local_weights[i].items():
                temp_dict[key] += value * self.len_dataset[i] / total_num

        return temp_dict

    def add_noise_to_aggregated_update(self, update_dict):
        if self.dp_noise_multiplier <= 0:
            return self.clone_parameter_dict(update_dict)

        noised = OrderedDict()
        std = self.dp_noise_multiplier * self.dp_clip_norm
        for idx, (name, value) in enumerate(update_dict.items()):
            generator = torch.Generator(device=value.device)
            generator.manual_seed(self.dp_seed + idx)
            noise = torch.randn(
                value.shape,
                generator=generator,
                device=value.device,
                dtype=torch.float32,
            ).to(value.dtype) * std
            noised[name] = value.data.clone() + noise
        return noised

    def set_global_parameter(self, para):
        state_dict = self.model.state_dict()
        for key, value in para.items():
            state_dict[key] = value.data.clone()
        self.model.load_state_dict(state_dict)

    def send_global_parameter(self, para):
        for client in self.clients:
            client.set_parameters(para)

    def evaluate(self, r):
        if self.dataset == 'ucf':
            gt = np.load("./data/gt_ucf.npy")
        else:
            gt = np.load("./data/gt_xd.npy")

        roc, ap = inference(self.dataset, self.model,
                            self.test_loader, gt, self.device)
        print(f"round {r + 1} : roc: {roc} , ap: {ap}")
        res_dict = {
            'ucf': roc,
            'xd': ap
        }
        return res_dict

    def evaluate_local(self, r):
        if self.dataset == 'ucf':
            gt = np.load("./data/gt_ucf.npy")
        else:
            gt = np.load("./data/gt_xd.npy")

        res_dict = {
            'ucf': [],
            'xd': []
        }
        for index in range(len(self.local_weights)):
            self.set_global_parameter(self.local_weights[index])
            roc, ap = inference(self.dataset, self.model,
                                self.test_loader, gt, self.device)
            res_dict['ucf'].append(roc)
            res_dict['xd'].append(ap)

        print(f"round {r + 1} : roc: {sum(res_dict['ucf']) / len(res_dict['ucf'])} ,"
              f" ap: {sum(res_dict['ap']) / len(res_dict['ap'])}")

    def train(self, dir_name):
        # 早停法参数
        patience = 10  # 连续10轮无改善则停止
        min_improvement = 0.0001  # 0.01%的最小改善阈值
        no_improvement_count = 0  # 记录连续无改善的轮数
        
        for g in range(self.global_rounds):
            print(f"\n-------- round: {g + 1} / {self.global_rounds} --------")
            self.local_weights.clear()
            self.local_losses.clear()
            self.len_dataset.clear()
            local_dp_stats = []

            i = 0
            for client in self.clients:
                i += 1
                print(f"round {g + 1}, client: {i}")
                w, loss, l_data, dp_stats = client.train()
                if self.use_dp and dp_stats is not None:
                    local_dp_stats.append(dp_stats)

                self.local_weights.append(w)
                self.local_losses.append(loss)
                self.len_dataset.append(l_data)
                client.scheduler.step()
                print()  # 每个客户端训练完成后添加空行

            if self.use_dp:
                aggregated_update = self.aggregate_updates()
                if self.dp_noise_mode == "central":
                    aggregated_update = self.add_noise_to_aggregated_update(aggregated_update)
                self.global_parameter = self.add_parameter_dict(self.global_parameter, aggregated_update)
                if self.dp_log_norm_stats and local_dp_stats:
                    avg_raw_norm = sum(item["raw_update_norm"] for item in local_dp_stats) / len(local_dp_stats)
                    avg_clipped_norm = sum(item["clipped_update_norm"] for item in local_dp_stats) / len(local_dp_stats)
                    avg_clip_coef = sum(item["clip_coef"] for item in local_dp_stats) / len(local_dp_stats)
                    print(
                        f"DP stats round {g + 1}: "
                        f"raw_norm={avg_raw_norm:.6f}, "
                        f"clipped_norm={avg_clipped_norm:.6f}, "
                        f"clip_coef={avg_clip_coef:.6f}"
                    )
            else:
                self.global_parameter = self.aggregate_parameters()
            self.set_global_parameter(self.global_parameter)
            res = self.evaluate(g)

            self.send_global_parameter(self.global_parameter)
            metric_name = 'roc' if self.dataset == 'ucf' else 'ap'
            current_metric = res[self.dataset]

            # 检查是否有改善（当前指标 > 最佳指标 * (1 + 最小改善阈值)）
            improvement_threshold = self.best * (1 + min_improvement)
            has_improvement = current_metric > improvement_threshold

            if has_improvement:
                self.best = current_metric
                self.best_model = self.global_parameter
                model_path = f"{dir_name}/model_{metric_name}_{current_metric:.4f}.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"\n模型已保存到: {model_path}")
                print(f"当前最佳{metric_name.upper()}: {self.best:.4f}")
                no_improvement_count = 0  # 重置计数器
            else:
                no_improvement_count += 1
                print(f"\n无显著改善 ({no_improvement_count}/{patience})")

            print(f"\n当前轮次{metric_name.upper()}: {current_metric:.4f}")
            print(f"历史最佳{metric_name.upper()}: {self.best:.4f}")
            print(f"连续无改善轮数: {no_improvement_count}/{patience}")
            print("\n" + "="*50)  # 添加分隔线

            # 早停判断
            if no_improvement_count >= patience:
                print(f"\n早停触发！连续{patience}轮无显著改善（阈值: {min_improvement*100:.2f}%）")
                print(f"在第 {g + 1} 轮停止训练")
                break

        print("\n训练完成！")
        final_model_path = f"{dir_name}/model_final_{metric_name}_{self.best:.4f}.pth"
        self.set_global_parameter(self.best_model)
        torch.save(self.model.state_dict(), final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
        print(f"最终{metric_name.upper()}: {self.best:.4f}")
        
        # 返回最佳评估指标
        return self.best
