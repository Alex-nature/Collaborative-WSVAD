import copy
from collections import OrderedDict
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
                 ):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.learning_rate = learning_rate
        self.train_loaders = train_loaders
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.label_map = label_map
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)
        self.scheduler = MultiStepLR(
            self.optimizer, scheduler_milestones, scheduler_rate)

    def set_parameters(self, new_params):
        state_dict = self.model.state_dict()
        for key, value in new_params.items():
            state_dict[key] = value.data.clone()
        self.model.load_state_dict(state_dict)

    def get_global_parameters(self):
        new_parameters = OrderedDict()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                new_parameters[name] = p.data.clone()
        return new_parameters

    def train(self):
        self.model.train()
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
                    # 文本特征正则化
                    loss_text_reg = text_branch_regularization(text_features_pos, text_features_neg)

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

            return (self.get_global_parameters(), loss_total2,
                    len(self.train_loaders[0]) + len(self.train_loaders[1]))

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
                    # 文本特征正则化
                    loss_text_reg = text_branch_regularization(text_features_pos, text_features_neg)

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

            return (self.get_global_parameters(), loss_total2,
                    len(self.train_loaders))