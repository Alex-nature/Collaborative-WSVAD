import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from utils import config
from utils.dataset import XDDataset, UCFDataset
from utils.model import Model
from utils.tools import get_prompt_text, build_negative_prompts


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
    neg_prompt_text, _ = build_negative_prompts(label_map)
    # neg_prompt_text = None
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
            outputs = model(visual, prompt_text, lengths, is_training=False, neg_text=neg_prompt_text)
            if isinstance(outputs, tuple):
                logits2, logits2_neg = outputs
            else:
                logits2, logits2_neg = outputs, None

            # 展平时间维
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            if logits2_neg is not None:
                logits2_neg = logits2_neg.reshape(logits2_neg.shape[0] * logits2_neg.shape[1], logits2_neg.shape[2])

            # 正分支获取 normal 置信度 -> 1 - P(normal)
            # ===== scores =====
            pos_logits = logits2[0:len_cur]  # [L, C_pos]
            normal_idx = prompt_text.index('normal')

            # 正分支：1 - P_yes(normal)
            probs_pos = pos_logits.softmax(dim=-1)
            score_pos = 1.0 - probs_pos[:, normal_idx]     # [L]

            # 负分支：用 normal vs abnormal 的二选一（不要对全类 softmax）
            if logits2_neg is not None:
                neg_logits = logits2_neg[0:len_cur]        # [L, C_neg]
                abnormal_idx = neg_prompt_text.index('abnormal')

                # tau 可调（建议 5~20），防止 sigmoid 饱和
                tau = 10.0
                score_neg = torch.sigmoid((neg_logits[:, abnormal_idx] - pos_logits[:, normal_idx]) / tau)  # [L]
            else:
                score_neg = torch.zeros_like(score_pos)

            # 融合：alpha 可调（0~1）
            alpha = 0.9
            fused_score = alpha * score_pos + (1.0 - alpha) * score_neg  # [L]
            prob2 = fused_score


            prob2 = fused_score.squeeze(-1)

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