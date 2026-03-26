import copy
from collections import OrderedDict
import math
import torch
from torch.optim.lr_scheduler import MultiStepLR
from utils.tools import (
    get_batch_label,
    get_prompt_text,
    CLASM,
    NEG_LOSS_BCE,
    build_negative_prompts,
    text_branch_regularization,
)
from tqdm import tqdm
import os


class FedAvgClient:
    def __init__(self,
                 model,
                 learning_rate: float,
                 train_loaders: tuple,
                 dataset: str,
                 local_epochs: int,
                 label_map,
                 scheduler_milestones,
                 scheduler_rate,
                 device: str,
                 use_dp: bool = False,
                 dp_clip_norm: float = 1.0,
                 dp_noise_multiplier: float = 0.0,
                 dp_seed: int = 20260326,
                 ):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.learning_rate = learning_rate
        self.train_loaders = train_loaders
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.label_map = label_map
        self.device = device
        self.use_dp = use_dp
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_seed = dp_seed

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)
        self.scheduler = MultiStepLR(
            self.optimizer, scheduler_milestones, scheduler_rate)
        self.global_parameters_snapshot = None

    @staticmethod
    def clone_parameter_dict(parameter_dict):
        cloned = OrderedDict()
        for name, value in parameter_dict.items():
            cloned[name] = value.data.clone()
        return cloned

    @staticmethod
    def subtract_parameter_dict(minuend, subtrahend):
        diff = OrderedDict()
        for name in minuend.keys():
            diff[name] = minuend[name].data.clone() - subtrahend[name].data.clone()
        return diff

    def set_parameters(self, new_params):
        state_dict = self.model.state_dict()
        for key, value in new_params.items():
            state_dict[key] = value.data.clone()
        self.model.load_state_dict(state_dict)
        self.global_parameters_snapshot = self.clone_parameter_dict(new_params)

    def get_global_parameters(self):
        new_parameters = OrderedDict()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                new_parameters[name] = p.data.clone()
        return new_parameters

    def get_model_update(self):
        if self.global_parameters_snapshot is None:
            raise RuntimeError("Global parameter snapshot is not set before local training.")
        local_parameters = self.get_global_parameters()
        return self.subtract_parameter_dict(local_parameters, self.global_parameters_snapshot)

    @staticmethod
    def parameter_dict_l2_norm(parameter_dict):
        squared_norm = 0.0
        for value in parameter_dict.values():
            squared_norm += torch.sum(value.detach().float() ** 2).item()
        return math.sqrt(squared_norm)

    @staticmethod
    def scale_parameter_dict(parameter_dict, scale):
        scaled = OrderedDict()
        for name, value in parameter_dict.items():
            scaled[name] = value.data.clone() * scale
        return scaled

    @staticmethod
    def add_gaussian_noise(parameter_dict, std, seed=None):
        noised = OrderedDict()
        for idx, (name, value) in enumerate(parameter_dict.items()):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=value.device)
                generator.manual_seed(seed + idx)
            noise = torch.randn(
                value.shape,
                generator=generator,
                device=value.device,
                dtype=torch.float32,
            ).to(value.dtype) * std
            noised[name] = value.data.clone() + noise
        return noised

    def apply_dp_to_update(self, model_update, clip_norm, noise_multiplier, generator=None):
        raw_norm = self.parameter_dict_l2_norm(model_update)
        clip_coef = min(1.0, clip_norm / (raw_norm + 1e-12))
        clipped_update = self.scale_parameter_dict(model_update, clip_coef)
        clipped_norm = self.parameter_dict_l2_norm(clipped_update)

        if noise_multiplier > 0:
            noised_update = self.add_gaussian_noise(
                clipped_update,
                std=noise_multiplier * clip_norm,
                seed=self.dp_seed,
            )
        else:
            noised_update = self.clone_parameter_dict(clipped_update)

        return noised_update, {
            "raw_update_norm": raw_norm,
            "clipped_update_norm": clipped_norm,
            "clip_coef": clip_coef,
        }

    def train(self):
        self.model.train()
        dp_stats = None
        prompt_text = get_prompt_text(self.label_map)
        # 构造负分支类别
        neg_prompt_text, neg_label_map = build_negative_prompts(self.label_map)

        if self.dataset == 'ucf':
            loss_total2 = 0
            epoch_losses = []       # 平均总loss
            epoch_ce_losses = []    # 平均正分支CE
            epoch_neg_losses = []   # 平均负分支VTOM
            epoch_text_reg_losses = []   # 平均文本正则化

            for epoch in range(self.local_epochs):
                normal_iter = iter(self.train_loaders[0])
                anomaly_iter = iter(self.train_loaders[1])

                loss_sum_total = 0.0
                loss_sum_ce = 0.0
                loss_sum_neg = 0.0
                loss_sum_text_reg = 0.0
                iters = 0

                total_iters = min(len(self.train_loaders[0]), len(self.train_loaders[1]))

                try:
                    term_width = os.get_terminal_size().columns
                except (OSError, IOError):
                    term_width = 80

                pbar = tqdm(
                    range(total_iters),
                    desc=f'Epoch {epoch+1}/{self.local_epochs}',
                    total=total_iters,
                    bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                    ncols=term_width
                )

                for _ in pbar:
                    normal_features, normal_label, normal_lengths = next(normal_iter)
                    anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

                    visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(self.device)
                    text_labels_raw = list(normal_label) + list(anomaly_label)
                    feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(self.device)

                    text_labels = get_batch_label(
                        text_labels_raw, prompt_text, self.label_map, self.dataset
                    ).to(self.device)

                    # VTOM_multi_hot使用正分支标签来确定负类别，无需构造负分支标签

                    # 模型前向：正/负双分支
                    logits_pos, logits_neg, text_features_pos, text_features_neg = self.model(
                        visual_features,
                        prompt_text,
                        feat_lengths,
                        is_training=True,
                        neg_text=neg_prompt_text,
                        return_text_features=True,
                    )

                    # 正分支 CE (CLASM)
                    loss_ce = CLASM(logits_pos, text_labels, feat_lengths, self.device)
                    # 负分支 VTOM_multi_hot
                    loss_neg = NEG_LOSS_BCE(logits_pos, logits_neg, text_labels, feat_lengths, self.device)
                    # 文本特征正则化 (优化版：使用平滑激活函数和温度参数)
                    loss_text_reg = text_branch_regularization(text_features_pos, text_features_neg,
                                                             reg_lambda=0.01, temperature=0.1)

                    loss = loss_ce + loss_neg + loss_text_reg

                    loss_sum_total += loss.item()
                    loss_sum_ce += loss_ce.item()
                    loss_sum_neg += loss_neg.item()
                    loss_sum_text_reg += loss_text_reg.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    iters += 1
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                avg_loss_epoch = loss_sum_total / iters
                avg_ce_epoch = loss_sum_ce / iters
                avg_neg_epoch = loss_sum_neg / iters
                avg_text_reg_epoch = loss_sum_text_reg / iters

                epoch_losses.append(avg_loss_epoch)
                epoch_ce_losses.append(avg_ce_epoch)
                epoch_neg_losses.append(avg_neg_epoch)
                epoch_text_reg_losses.append(avg_text_reg_epoch)

                loss_total2 += avg_loss_epoch

                print(
                    f'  Epoch {epoch+1}/{self.local_epochs} '
                    f'平均Loss: {avg_loss_epoch:.6f} | '
                    f'CE: {avg_ce_epoch:.6f} | '
                    f'NEG_VTOM: {avg_neg_epoch:.6f} | '
                    f'TEXT_REG: {avg_text_reg_epoch:.6f}'
                )

            print(f'  所有Epoch Loss     : {[f"{loss:.6f}" for loss in epoch_losses]}')
            print(f'  所有Epoch CE      : {[f"{loss:.6f}" for loss in epoch_ce_losses]}')
            print(f'  所有Epoch NEG_VTOM : {[f"{loss:.6f}" for loss in epoch_neg_losses]}')
            print(f'  所有Epoch TEXT_REG: {[f"{loss:.6f}" for loss in epoch_text_reg_losses]}')
            print(f'  总体平均Loss: {loss_total2/self.local_epochs:.6f}')

            if self.use_dp:
                model_update = self.get_model_update()
                protected_update, dp_stats = self.apply_dp_to_update(
                    model_update,
                    clip_norm=self.dp_clip_norm,
                    noise_multiplier=self.dp_noise_multiplier,
                    generator=self.dp_generator,
                )
                return (protected_update, loss_total2,
                        len(self.train_loaders[0]) + len(self.train_loaders[1]), dp_stats)

            return (self.get_global_parameters(), loss_total2,
                    len(self.train_loaders[0]) + len(self.train_loaders[1]), dp_stats)

        elif self.dataset == 'xd':
            loss_total2 = 0
            epoch_losses = []
            epoch_ce_losses = []
            epoch_neg_losses = []   # 平均负分支VTOM
            epoch_text_reg_losses = []   # 平均文本正则化

            for epoch in range(self.local_epochs):
                loss_sum_total = 0.0
                loss_sum_ce = 0.0
                loss_sum_neg = 0.0
                loss_sum_text_reg = 0.0
                iters = 0

                try:
                    term_width = os.get_terminal_size().columns
                except (OSError, IOError):
                    term_width = 80

                total_batches = len(self.train_loaders)

                pbar = tqdm(
                    enumerate(self.train_loaders),
                    desc=f'Epoch {epoch+1}/{self.local_epochs}',
                    total=total_batches,
                    bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                    ncols=term_width
                )

                for _, item in pbar:
                    visual_feat, text_labels_raw, feat_lengths = item
                    visual_feat = visual_feat.to(self.device)
                    feat_lengths = feat_lengths.to(self.device)

                    text_labels = get_batch_label(
                        text_labels_raw, prompt_text, self.label_map, self.dataset
                    ).to(self.device)

                    # 模型前向：双分支推理
                    logits_pos, logits_neg, text_features_pos, text_features_neg = self.model(
                        visual_feat,
                        prompt_text,
                        feat_lengths,
                        is_training=True,
                        neg_text=neg_prompt_text,
                        return_text_features=True,
                    )

                    # 正分支 CE (CLASM)
                    loss_ce = CLASM(logits_pos, text_labels, feat_lengths, self.device)
                    # 负分支 VTOM_multi_hot
                    loss_neg = NEG_LOSS_BCE(logits_pos, logits_neg, text_labels, feat_lengths, self.device)
                    # 文本特征正则化 (优化版：使用平滑激活函数和温度参数)
                    loss_text_reg = text_branch_regularization(text_features_pos, text_features_neg,
                                                             reg_lambda=1.0, temperature=0.1)

                    loss = loss_ce + loss_neg + loss_text_reg

                    loss_sum_total += loss.item()
                    loss_sum_ce += loss_ce.item()
                    loss_sum_neg += loss_neg.item()
                    loss_sum_text_reg += loss_text_reg.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    iters += 1
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                avg_loss_epoch = loss_sum_total / iters
                avg_ce_epoch = loss_sum_ce / iters
                avg_neg_epoch = loss_sum_neg / iters
                avg_text_reg_epoch = loss_sum_text_reg / iters

                epoch_losses.append(avg_loss_epoch)
                epoch_ce_losses.append(avg_ce_epoch)
                epoch_neg_losses.append(avg_neg_epoch)
                epoch_text_reg_losses.append(avg_text_reg_epoch)

                loss_total2 += avg_loss_epoch

                print(
                    f'  Epoch {epoch+1}/{self.local_epochs} '
                    f'平均Loss: {avg_loss_epoch:.6f} | '
                    f'CE: {avg_ce_epoch:.6f} | '
                    f'NEG_VTOM: {avg_neg_epoch:.6f} | '
                    f'TEXT_REG: {avg_text_reg_epoch:.6f}'
                )

            print(f'  所有Epoch Loss     : {[f"{loss:.6f}" for loss in epoch_losses]}')
            print(f'  所有Epoch CE      : {[f"{loss:.6f}" for loss in epoch_ce_losses]}')
            print(f'  所有Epoch NEG_VTOM : {[f"{loss:.6f}" for loss in epoch_neg_losses]}')
            print(f'  所有Epoch TEXT_REG: {[f"{loss:.6f}" for loss in epoch_text_reg_losses]}')
            print(f'  总体平均Loss: {loss_total2/self.local_epochs:.6f}')

            if self.use_dp:
                model_update = self.get_model_update()
                protected_update, dp_stats = self.apply_dp_to_update(
                    model_update,
                    clip_norm=self.dp_clip_norm,
                    noise_multiplier=self.dp_noise_multiplier,
                    generator=self.dp_generator,
                )
                return (protected_update, loss_total2,
                        len(self.train_loaders), dp_stats)

            return (self.get_global_parameters(), loss_total2,
                    len(self.train_loaders), dp_stats)
