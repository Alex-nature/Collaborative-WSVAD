import copy
from collections import OrderedDict
import torch
from torch.optim.lr_scheduler import MultiStepLR
from utils.tools import get_batch_label, get_prompt_text, CLASM, build_negative_prompts, VTOM
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

        # 负分支损失权重（可调）
        self.vto_weight = 0
        # 负分支温度参数（防止sigmoid饱和导致VTO过快趋近0）
        self.vto_tau = 10  # 常用 5~20；先用10比较稳

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
        # neg_prompt_text, _ = build_negative_prompts(self.label_map)
        neg_prompt_text = None

        if self.dataset == 'ucf':
            loss_total2 = 0
            epoch_losses = []       # 平均总loss（包含VTO）
            epoch_ce_losses = []    # 平均CE
            epoch_vto_losses = []   # 平均VTO

            for epoch in range(self.local_epochs):
                normal_iter = iter(self.train_loaders[0])
                anomaly_iter = iter(self.train_loaders[1])

                loss_sum_total = 0.0
                loss_sum_ce = 0.0
                loss_sum_vto = 0.0
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

                    logits_out = self.model(
                        visual_features,
                        prompt_text,
                        feat_lengths,
                        is_training=True,
                        neg_text=neg_prompt_text
                    )

                    if isinstance(logits_out, tuple):
                        logits_pos, logits_neg = logits_out
                    else:
                        logits_pos, logits_neg = logits_out, None

                    # CE
                    loss_ce = CLASM(logits_pos, text_labels, feat_lengths, self.device)

                    # VTO
                    if logits_neg is not None:
                        loss_vto = VTOM(
                            logits_pos, logits_neg, text_labels, feat_lengths, self.device,
                            tau=self.vto_tau
                        )
                        loss = loss_ce + self.vto_weight * loss_vto
                        vto_val = loss_vto.item()
                    else:
                        loss_vto = None
                        loss = loss_ce
                        vto_val = 0.0

                    loss_sum_total += loss.item()
                    loss_sum_ce += loss_ce.item()
                    loss_sum_vto += vto_val

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    iters += 1
                    if loss_vto is not None:
                        pbar.set_postfix({
                            'ce': f'{loss_ce.item():.4f}',
                            'vto': f'{loss_vto.item():.4f}',
                            'loss': f'{loss.item():.4f}'
                        })
                    else:
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                avg_loss_epoch = loss_sum_total / iters
                avg_ce_epoch = loss_sum_ce / iters
                avg_vto_epoch = loss_sum_vto / iters

                epoch_losses.append(avg_loss_epoch)
                epoch_ce_losses.append(avg_ce_epoch)
                epoch_vto_losses.append(avg_vto_epoch)

                loss_total2 += avg_loss_epoch

                print(
                    f'  Epoch {epoch+1}/{self.local_epochs} '
                    f'平均Loss: {avg_loss_epoch:.6f} | '
                    f'CE: {avg_ce_epoch:.6f} | '
                    f'VTO: {avg_vto_epoch:.6f}'
                )

            print(f'  所有Epoch Loss: {[f"{loss:.6f}" for loss in epoch_losses]}')
            print(f'  所有Epoch CE  : {[f"{loss:.6f}" for loss in epoch_ce_losses]}')
            print(f'  所有Epoch VTO : {[f"{loss:.6f}" for loss in epoch_vto_losses]}')
            print(f'  总体平均Loss: {loss_total2/self.local_epochs:.6f}')

            return (self.get_global_parameters(), loss_total2,
                    len(self.train_loaders[0]) + len(self.train_loaders[1]))

        elif self.dataset == 'xd':
            loss_total2 = 0
            epoch_losses = []
            epoch_ce_losses = []
            epoch_vto_losses = []

            for epoch in range(self.local_epochs):
                loss_sum_total = 0.0
                loss_sum_ce = 0.0
                loss_sum_vto = 0.0
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

                    logits_out = self.model(
                        visual_feat,
                        prompt_text,
                        feat_lengths,
                        is_training=True,
                        neg_text=neg_prompt_text
                    )

                    if isinstance(logits_out, tuple):
                        logits_pos, logits_neg = logits_out
                    else:
                        logits_pos, logits_neg = logits_out, None

                    loss_ce = CLASM(logits_pos, text_labels, feat_lengths, self.device)

                    if logits_neg is not None:
                        loss_vto = VTOM(
                            logits_pos, logits_neg, text_labels, feat_lengths, self.device,
                            tau=self.vto_tau
                        )
                        loss = loss_ce + self.vto_weight * loss_vto
                        vto_val = loss_vto.item()
                    else:
                        loss_vto = None
                        loss = loss_ce
                        vto_val = 0.0

                    loss_sum_total += loss.item()
                    loss_sum_ce += loss_ce.item()
                    loss_sum_vto += vto_val

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    iters += 1
                    if loss_vto is not None:
                        pbar.set_postfix({
                            'ce': f'{loss_ce.item():.4f}',
                            'vto': f'{loss_vto.item():.4f}',
                            'loss': f'{loss.item():.4f}'
                        })
                    else:
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                avg_loss_epoch = loss_sum_total / iters
                avg_ce_epoch = loss_sum_ce / iters
                avg_vto_epoch = loss_sum_vto / iters

                epoch_losses.append(avg_loss_epoch)
                epoch_ce_losses.append(avg_ce_epoch)
                epoch_vto_losses.append(avg_vto_epoch)

                loss_total2 += avg_loss_epoch

                print(
                    f'  Epoch {epoch+1}/{self.local_epochs} '
                    f'平均Loss: {avg_loss_epoch:.6f} | '
                    f'CE: {avg_ce_epoch:.6f} | '
                    f'VTO: {avg_vto_epoch:.6f}'
                )

            print(f'  所有Epoch Loss: {[f"{loss:.6f}" for loss in epoch_losses]}')
            print(f'  所有Epoch CE  : {[f"{loss:.6f}" for loss in epoch_ce_losses]}')
            print(f'  所有Epoch VTO : {[f"{loss:.6f}" for loss in epoch_vto_losses]}')
            print(f'  总体平均Loss: {loss_total2/self.local_epochs:.6f}')

            return (self.get_global_parameters(), loss_total2,
                    len(self.train_loaders))