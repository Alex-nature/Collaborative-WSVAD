import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from utils import config
from utils.dataset import XDDataset, UCFDataset
from utils.model import Model
from utils.tools import get_prompt_text, get_negative_prompt_text_ucf


def dynamic_weighted_fusion(anomaly_pos, anomaly_neg, probs_pos, probs_neg,
                          normal_idx, abnormal_idx, consistency_boost=0.3):
    """
    动态权重融合：基于置信度和一致性自适应调整融合权重

    Args:
        anomaly_pos/neg: 正负分支的异常分数
        probs_pos/neg: 正负分支的概率分布
        normal_idx: normal类别的索引
        abnormal_idx: abnormal类别的索引
        consistency_boost: 一致性增强系数

    Returns:
        fused_anomaly: 融合后的异常分数
        pos_weight: 正分支权重
        neg_weight: 负分支权重
    """
    # 1. 计算置信度
    pos_confidence = torch.abs(probs_pos[:, normal_idx] - 0.5) * 2
    neg_confidence = torch.abs(probs_neg[:, abnormal_idx] - 0.5) * 2

    # 2. 计算一致性
    diff = torch.abs(anomaly_pos - anomaly_neg)
    consistency = 1 - diff

    # 3. 计算动态权重
    base_pos_weight = 0.7
    base_neg_weight = 0.3

    pos_weight = pos_confidence * (base_pos_weight + consistency_boost * consistency)
    neg_weight = neg_confidence * (base_neg_weight + consistency_boost * consistency)

    # 4. 归一化权重
    total_weight = pos_weight + neg_weight
    pos_weight = pos_weight / total_weight
    neg_weight = neg_weight / total_weight

    # 5. 加权融合
    fused_anomaly = pos_weight * anomaly_pos + neg_weight * anomaly_neg

    return fused_anomaly, pos_weight, neg_weight


def inference(dataset, model, test_loader, gt, device):
    visual_length = 256
    model.eval()
    model.to(device)
    if dataset == 'ucf':
        label_map = dict(
            {'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
             'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
             'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
             'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})
    else:
        label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
                          'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    prompt_text = get_prompt_text(label_map)
    neg_prompt_text = get_negative_prompt_text_ucf(prompt_text)
    with torch.no_grad():
        max_len = 256
        for i, item in enumerate(test_loader):

            visual = item[0].squeeze(0)
            visual = visual.to(device)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < visual_length:
                visual = visual.unsqueeze(0)

            lengths = torch.zeros(int(length / max_len) + 1)

            for j in range(int(length / max_len) + 1):
                if j == 0 and length < max_len:
                    lengths[j] = length
                elif j == 0 and length > max_len:
                    lengths[j] = max_len
                    length -= max_len
                elif length > max_len:
                    lengths[j] = max_len
                    length -= max_len
                else:
                    lengths[j] = length
            lengths = lengths.to(int)

            # 获取正负分支logits进行融合推理
            logits_pos, logits_neg = model(visual, prompt_text, lengths, is_training=True, neg_text=neg_prompt_text)

            # 展平时间维
            logits_pos = logits_pos.reshape(logits_pos.shape[0] * logits_pos.shape[1], logits_pos.shape[2])
            logits_neg = logits_neg.reshape(logits_neg.shape[0] * logits_neg.shape[1], logits_neg.shape[2])

            # 计算异常分数
            probs_pos = logits_pos[0:len_cur].softmax(dim=-1)  # 正分支：多类别
            probs_neg = logits_neg[0:len_cur].sigmoid()        # 负分支：多标签

            normal_idx = prompt_text.index('normal')
            abnormal_idx = 0  # neg_prompt_text[0] = 'abnormal'

            anomaly_score_pos = (1 - probs_pos[:, normal_idx]).squeeze(-1)
            anomaly_score_neg = probs_neg[:, abnormal_idx].squeeze(-1)

            # 动态权重融合
            prob2, pos_weight, neg_weight = dynamic_weighted_fusion(
                anomaly_score_pos, anomaly_score_neg, probs_pos, probs_neg,
                normal_idx, abnormal_idx
            )

            if i == 0:
                ap2 = prob2
            else:
                ap2 = torch.cat([ap2, prob2], dim=0)

    ap2 = ap2.cpu().numpy()
    ap2 = ap2.tolist()
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    return ROC2, AP2



if __name__ == "__main__":
    args = config.parser.parse_args()

    if args.dataset == 'ucf':
        test_list = './data/list/ucf_test.csv'
        test_dataset = UCFDataset(args.visual_length, test_list, True)

    else:
        test_list = './data/list/xd_test.csv'
        test_dataset = XDDataset(args.visual_length, test_list, True)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')

    # ！！！# If the checkpoint contains a 'model_state_dict' key, use it; otherwise, use the entire checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    roc, ap = 0, 0
    if args.dataset == 'ucf':
        gt = np.load("./data/gt_ucf.npy")
        roc, ap = inference('ucf', model, test_loader, gt, device)
    elif args.dataset == 'xd':
        gt = np.load("./data/gt_xd.npy")
        roc, ap = inference('xd', model, test_loader, gt, device)

    print(f'roc: {roc}, ap: {ap}')