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


def get_negative_prompt_text_ucf(pos_prompt_text):
    """
    根据 UCF 正分支的 prompt_text 构造负分支的类别名列表。

    约定:
        pos_prompt_text: ['normal', 'abuse', 'arrest', ..., 'vandalism']
        返回:
        neg_prompt_text: ['abnormal', 'no_abuse', 'no_arrest', ..., 'no_vandalism']
    """
    neg_prompt_text = ['abnormal']
    for name in pos_prompt_text:
        if name == 'normal':
            continue
        neg_prompt_text.append(f'no_{name}')
    return neg_prompt_text


def get_negative_prompt_text_xd(pos_prompt_text):
    """
    根据 XD 正分支的 prompt_text 构造负分支的类别名列表。

    约定:
        pos_prompt_text: ['normal', 'fighting', 'shooting', 'riot', 'abuse', 'car accident', 'explosion']
        返回:
        neg_prompt_text: ['abnormal', 'no_fighting', 'no_shooting', 'no_riot', 'no_abuse', 'no_car accident', 'no_explosion']
    """
    neg_prompt_text = ['abnormal']
    for name in pos_prompt_text:
        if name == 'normal':
            continue
        neg_prompt_text.append(f'no_{name}')
    return neg_prompt_text


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


def NEG_LOSS_BCE(logits, labels, lengths, device):
    """
    负分支专用损失函数：
    - 与 CLASM 一样，在时间维做 top-k 聚合得到视频级 logits
    - 但在类别维度使用逐类二元交叉熵（BCE），不做 softmax

    参数:
        logits: [B, T, C_neg]  负分支的时序 logits
        labels: [B, C_neg]     0/1 标签，表示每个负类对该视频是否“成立”
        lengths: [B]           每个视频的有效长度
        device: 设备字符串
    """
    instance_logits = torch.zeros(0).to(device)  # [0, C_neg]
    labels = labels.to(device).float()

    for i in range(logits.shape[0]):
        valid_len = int(lengths[i].item())
        # 防止长度太短导致 k=0，至少取 1
        k = max(int(valid_len / 16 + 1), 1)
        # 只在有效帧上做 top-k
        tmp, _ = torch.topk(logits[i, 0:valid_len], k=k, largest=True, dim=0)  # [k, C_neg]
        pooled = torch.mean(tmp, 0, keepdim=True)  # [1, C_neg]
        instance_logits = torch.cat([instance_logits, pooled], dim=0)

    loss = F.binary_cross_entropy_with_logits(instance_logits, labels)
    return loss


def get_batch_negative_label_ucf(texts, pos_prompt_text, neg_prompt_text, label_map):
    """
    为 UCF 构造负分支标签:
        neg_prompt_text: ['abnormal', 'no_abuse', 'no_arrest', ..., 'no_vandalism']
    规则:
        - Normal 视频:
            abnormal = 0
            所有 no_xxx = 1
        - 异常 e0 视频:
            abnormal = 1
            对发生的异常 e0:        no_e0 = 0
            对未发生的其它异常 e:   no_e = 1
    """
    label_vectors = torch.zeros(0)

    # 提取事件类名列表（不含 normal），并保证顺序与 neg_prompt_text 中一致
    event_names = [name for name in pos_prompt_text if name != 'normal']

    for text in texts:
        label_vector = torch.zeros(len(neg_prompt_text))

        if text == 'Normal':
            # normal 视频
            # abnormal = 0
            label_vector[0] = 0
            # 所有 no_xxx = 1
            label_vector[1:] = 1
        else:
            # 单一异常事件
            if text in label_map:
                e_name = label_map[text]  # 映射到小写事件名，如 'Fighting' -> 'fighting'
            else:
                # 若找不到映射，保守起见视为 abnormal 且无特定事件
                e_name = None

            # abnormal = 1
            label_vector[0] = 1

            # 遍历事件名，对应到 neg_prompt_text 中的 no_xxx
            for idx, ev_name in enumerate(event_names, start=1):
                if e_name is not None and ev_name == e_name:
                    # 实际发生的事件，其 no_xxx 应为 0
                    label_vector[idx] = 0
                else:
                    # 未发生的其它事件，其 no_xxx 为 1
                    label_vector[idx] = 1

        label_vector = label_vector.unsqueeze(0)
        label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors


def get_batch_negative_label_xd(texts, pos_prompt_text, neg_prompt_text, label_map):
    """
    为 XD 构造负分支标签:
        neg_prompt_text: ['abnormal', 'no_fighting', 'no_shooting', 'no_riot', 'no_abuse', 'no_car accident', 'no_explosion']
    规则:
        - Normal 视频 (标签 'A'):
            abnormal = 0
            所有 no_xxx = 1
        - 异常 e0 视频 (标签 'B1', 'B2', 'B4', 'B5', 'B6', 'G'):
            abnormal = 1
            对发生的异常 e0:        no_e0 = 0
            对未发生的其它异常 e:   no_e = 1
    """
    label_vectors = torch.zeros(0)

    # 提取事件类名列表（不含 normal），并保证顺序与 neg_prompt_text 中一致
    event_names = [name for name in pos_prompt_text if name != 'normal']

    for text in texts:
        label_vector = torch.zeros(len(neg_prompt_text))

        if text == 'A':  # XD数据集中的正常样本标签
            # normal 视频
            # abnormal = 0
            label_vector[0] = 0
            # 所有 no_xxx = 1
            label_vector[1:] = 1
        else:
            # 单一异常事件
            if text in label_map:
                e_name = label_map[text]  # 映射到事件名，如 'B1' -> 'fighting'
            else:
                # 若找不到映射，保守起见视为 abnormal 且无特定事件
                e_name = None

            # abnormal = 1
            label_vector[0] = 1

            # 遍历事件名，对应到 neg_prompt_text 中的 no_xxx
            for idx, ev_name in enumerate(event_names, start=1):
                if e_name is not None and ev_name == e_name:
                    # 实际发生的事件，其 no_xxx 应为 0
                    label_vector[idx] = 0
                else:
                    # 未发生的其它事件，其 no_xxx 为 1
                    label_vector[idx] = 1

        label_vector = label_vector.unsqueeze(0)
        label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors