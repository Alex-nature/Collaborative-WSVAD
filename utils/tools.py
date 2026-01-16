import torch
import numpy as np
import torch.nn.functional as F


def get_batch_label(texts, prompt_text, label_map: dict, dataset):
    label_vectors = torch.zeros(0)
    if dataset == 'ucf':
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    # xd
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1

            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors


def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text


# 构造负提示：按已有 label_map 的 value 生成对应否定文本
def build_negative_prompts(label_map: dict):
    """
    输入: label_map (如 {"Abuse": "abuse", "Normal": "normal"})
    输出:
        neg_prompt_text: 与 label_map 顺序一致的负提示列表
        neg_label_map: {原始key: 负提示字符串}
    规则:
        - 对 normal/normal-like 使用 "abnormal"
        - 其他类别使用 "no {label_value}"
    """
    neg_prompt_text = []
    neg_label_map = {}

    for k, v in label_map.items():
        v_lower = v.lower()
        if v_lower in ["normal"]:
            neg_text = "abnormal"
        else:
            neg_text = f"no {v_lower}"
        neg_prompt_text.append(neg_text)
        neg_label_map[k] = neg_text

    return neg_prompt_text, neg_label_map

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1

    return mask.bool()


def random_extract(feat, t_max):
    r = np.random.randint(feat.shape[0] - t_max)
    return feat[r: r + t_max, :]


def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    if avg is True:
        for i in range(t_max):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
    else:
        r = np.linspace(0, feat.shape[0] - 1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]

    return new_feat


def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length


def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate(
                    [split_feat, feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate(
                    [split_feat,
                     pad(feat[i * length:i * length + length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss



def VTOM_multi_hot(
    logits_yes: torch.Tensor,  # [B, T, C]
    logits_no: torch.Tensor,   # [B, T, C]
    labels: torch.Tensor,      # [B, C]  one-hot (UCF) or multi-hot (XD)
    lengths: torch.Tensor,     # [B]
    device,
    eps: float = 1e-6,
    tau: float = 10.0,
):
    """
    Multi-hot generalized VTO loss (B plan):
    - P_no = sigmoid((logits_no - logits_yes)/tau)
    - low-K MIL pooling over time
    - penalize only negative classes (classes not present in labels)
    - normalize by #negative classes per sample (C - #pos), which reduces to (C-1) for one-hot.
    """
    logits_yes = logits_yes.to(device)
    logits_no  = logits_no.to(device)
    labels     = labels.to(device)

    # Binary positives for mask logic (IMPORTANT for multi-hot)
    labels_bin = (labels > 0).float()          # [B, C]
    neg_mask   = 1.0 - labels_bin              # [B, C]

    # Eq.(9) stable form (with tau softening)
    P_no = torch.sigmoid((logits_no - logits_yes) / tau)  # [B, T, C]

    B, T, C = P_no.shape
    lengths_cpu = lengths.detach().to("cpu")

    # low-K pooling per sample
    pno_list = []
    for i in range(B):
        L = int(lengths_cpu[i])
        L = max(1, min(L, T))                 # guard
        K = int(L / 16 + 1)                   # 使用给定的k值
        lowk_vals, _ = torch.topk(P_no[i, :L], k=K, largest=False, dim=0)  # [K, C]
        pno_list.append(lowk_vals.mean(dim=0))                              # [C]

    instance_pno = torch.stack(pno_list, dim=0)  # [B, C]

    # loss: only negatives contribute
    log_pno = torch.log(instance_pno.clamp_min(eps))  # [B, C]

    # normalize by number of negatives per sample: C - #pos
    denom = neg_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]

    loss_per_sample = -((neg_mask * log_pno).sum(dim=1, keepdim=True) / denom)  # [B, 1]

    return loss_per_sample.mean()


def NEG_LOSS_BCE(logits_pos, logits_neg, labels, lengths, device, tau=10.0):
    """
    负分支专用损失函数：基于VTOM_multi_hot的改进版本

    参数:
        logits_pos: [B, T, C]  正分支的时序 logits
        logits_neg: [B, T, C]  负分支的时序 logits
        labels: [B, C]         标签（正分支标签，用于确定负类别）
        lengths: [B]           每个视频的有效长度
        device: 设备
        tau: 温度参数
    """
    return VTOM_multi_hot(logits_pos, logits_neg, labels, lengths, device, tau=tau)


def text_branch_regularization(T_yes: torch.Tensor, T_no: torch.Tensor, reg_lambda: float = 0.1, temperature: float = 0.1) -> torch.Tensor:
    """
    TO loss first term: 正负分支文本特征余弦相似度正则化 (优化版)

    使用平滑激活函数和温度参数的组合优化版本

    公式: λ × (1/C) × Σᵢ softplus( cos(T_yes[i], T_no[i]) / τ )

    Args:
        T_yes: [C, D] tensor, yes text embeddings (one per class)
        T_no : [C, D] tensor, no  text embeddings (one per class)
        reg_lambda: 正则化强度系数 (默认0.1)
        temperature: 温度参数，控制相似度敏感度 (默认0.1)

    Returns:
        scalar tensor
    """
    if T_yes.shape != T_no.shape:
        raise ValueError(f"T_yes and T_no must have same shape, got {T_yes.shape} vs {T_no.shape}")
    if T_yes.dim() != 2:
        raise ValueError(f"Expected [C, D], got dim={T_yes.dim()}")

    # cosine similarity for each class i (paired row-wise)
    cos_sim = F.cosine_similarity(T_yes, T_no, dim=1)  # [C]

    # 温度缩放：控制相似度对角度变化的敏感度
    scaled_sim = cos_sim / temperature

    # 使用Softplus替代ReLU：平滑的近似ReLU，避免梯度消失
    loss = F.softplus(scaled_sim, beta=1.0).mean()  # mean over C equals (1/C) sum

    return reg_lambda * loss