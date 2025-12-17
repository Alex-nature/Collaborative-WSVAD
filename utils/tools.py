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


def VTOM(logits_yes, logits_no, labels, lengths, device, eps=1e-6, tau=10.0, debug=False):
    """
    logits_yes: [B, T, C]  正分支 logits (normal, abuse, fighting, ...)
    logits_no : [B, T, C]  负分支 logits (abnormal, no abuse, no fighting, ...)
    labels    : [B, C]     one-hot (仍按CLASM做归一化以对齐)
    lengths   : [B]
    tau       : temperature for sigmoid input scaling (avoid saturation). Try 5~20.
    debug     : if True, print delta / P_no stats occasionally.
    """
    # 与 CLASM 完全一致：先归一化标签（one-hot 情况下等价于不变）
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    # 关键改动：加入温度 tau，避免 sigmoid 过快饱和
    # delta = (logits_no - logits_yes) / tau
    # P_no = torch.sigmoid(delta)  # [B, T, C]
    delta = (logits_no - logits_yes) / tau
    P_no = torch.sigmoid(delta)
    # print("tau:", tau,
    #     "delta std:", delta.std().item(),
    #     "P_no mean:", P_no.mean().item(),
    #     "P_no min:", P_no.min().item())


    if debug:
        # 只打印一些统计值，方便看是否还饱和（P_no 是否几乎全是 0/1）
        with torch.no_grad():
            print(f"[VTOM debug] delta: mean={delta.mean().item():.3f}, std={delta.std().item():.3f}, "
                  f"min={delta.min().item():.3f}, max={delta.max().item():.3f}")
            print(f"[VTOM debug] P_no : mean={P_no.mean().item():.3f}, std={P_no.std().item():.3f}, "
                  f"min={P_no.min().item():.3f}, max={P_no.max().item():.3f}")

    instance_pno = torch.zeros((0, P_no.size(-1)), device=device)  # [B, C]
    for i in range(P_no.size(0)):
        L = int(lengths[i])
        K = int(L / 16 + 1)  # 与 CLASM 完全一致
        tmp, _ = torch.topk(P_no[i, 0:L], k=K, largest=False, dim=0)  # [K, C] low-K
        instance_pno = torch.cat([instance_pno, tmp.mean(dim=0, keepdim=True)], dim=0)

    neg_mask = 1.0 - labels  # [B, C]
    log_pno = torch.log(instance_pno.clamp(min=eps))  # 防止log(0)

    denom = neg_mask.sum(dim=1, keepdim=True).clamp(min=eps)
    milloss = -torch.mean(torch.sum(neg_mask * log_pno, dim=1, keepdim=True) / denom)

    return milloss
