import os
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from utils import config
from utils.dataset import UCFDataset, XDDataset, ShanghaiDataset
from utils.model import Model
from utils.tools import get_prompt_text, build_negative_prompts


def inference(dataset, model, test_loader, gt, device):
    visual_length = 256
    model.eval()
    model.to(device)

    if dataset == 'ucf':
        label_map = dict({
            'Normal': 'normal',
            'Abuse': 'abuse',
            'Arrest': 'arrest',
            'Arson': 'arson',
            'Assault': 'assault',
            'Burglary': 'burglary',
            'Explosion': 'explosion',
            'Fighting': 'fighting',
            'RoadAccidents': 'roadAccidents',
            'Robbery': 'robbery',
            'Shooting': 'shooting',
            'Shoplifting': 'shoplifting',
            'Stealing': 'stealing',
            'Vandalism': 'vandalism'
        })
    elif dataset == 'xd':
        label_map = dict({
            'A': 'normal',
            'B1': 'fighting',
            'B2': 'shooting',
            'B4': 'riot',
            'B5': 'abuse',
            'B6': 'car accident',
            'G': 'explosion'
        })
    elif dataset == 'shanghai':
        label_map = dict({
            'normal': 'normal',
            'vehicle': 'vehicle',
            'fighting': 'fighting',
            'skateboard': 'skateboard',
            'running': 'running',
            'robbery': 'robbery',
            'car': 'car',
            'fall': 'fall',
            'throwing_object': 'throwing object',
            'chasing': 'chasing',
            'jumping': 'jumping',
            'stoop': 'stoop',
            'push': 'push',
            'vaudeville': 'vaudeville',
            'monocycle': 'monocycle',
            'circuit': 'circuit',
            'hurdle': 'hurdle',
            'step': 'step'
        })
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    prompt_text = get_prompt_text(label_map)
    neg_prompt_text, _ = build_negative_prompts(label_map)

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

            tmp_len = length
            for j in range(int(length / max_len) + 1):
                if j == 0 and tmp_len < max_len:
                    lengths[j] = tmp_len
                elif j == 0 and tmp_len > max_len:
                    lengths[j] = max_len
                    tmp_len -= max_len
                elif tmp_len > max_len:
                    lengths[j] = max_len
                    tmp_len -= max_len
                else:
                    lengths[j] = tmp_len

            lengths = lengths.to(int)

            outputs = model(visual, prompt_text, lengths, is_training=False, neg_text=neg_prompt_text)
            if isinstance(outputs, tuple):
                logits2, logits2_neg = outputs
            else:
                logits2, logits2_neg = outputs, None

            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            if logits2_neg is not None:
                logits2_neg = logits2_neg.reshape(
                    logits2_neg.shape[0] * logits2_neg.shape[1],
                    logits2_neg.shape[2]
                )

            pos_logits = logits2[0:len_cur]
            normal_idx = prompt_text.index('normal')

            probs_pos = pos_logits.softmax(dim=-1)
            score_pos = 1.0 - probs_pos[:, normal_idx]

            if logits2_neg is not None:
                neg_logits = logits2_neg[0:len_cur]
                abnormal_idx = neg_prompt_text.index('abnormal')
                tau = 10.0
                score_neg = torch.sigmoid(
                    (neg_logits[:, abnormal_idx] - pos_logits[:, normal_idx]) / tau
                )
            else:
                score_neg = torch.zeros_like(score_pos)

            alpha = 1.0
            fused_score = alpha * score_pos + (1.0 - alpha) * score_neg
            prob2 = fused_score.squeeze(-1)

            if i == 0:
                ap2 = prob2
            else:
                ap2 = torch.cat([ap2, prob2], dim=0)

    ap2 = ap2.cpu().numpy().tolist()
    pred = np.repeat(ap2, 16)

    min_len = min(len(gt), len(pred))
    gt = gt[:min_len]
    pred = pred[:min_len]

    roc = roc_auc_score(gt, pred)
    ap = average_precision_score(gt, pred)

    return roc, ap


def test_one_crop_shanghai(args, model, gt, device, crop_id):
    test_list = rf"E:\codes\WSVAD_Datasets\shanghaitech\shanghaitech\testing\weakly setting\csv\stc_test_crop{crop_id}.csv"

    if not os.path.exists(test_list):
        print(f"[crop{crop_id}] csv不存在: {test_list}")
        return None

    test_dataset = ShanghaiDataset(args.visual_length, test_list, True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    roc, ap = inference('shanghai', model, test_loader, gt, device)
    return roc, ap


if __name__ == "__main__":
    args = config.parser.parse_args()

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
        use_tca=args.use_tca,
        tca_window_size=args.tca_window_size,
        tca_dropout=args.tca_dropout,
        use_distance_adj=args.use_distance_adj,
        tca_gamma=args.tca_gamma,
        tca_bias=args.tca_bias,
        tca_norm=args.tca_norm
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if args.dataset == 'ucf':
        test_list = './data/list/ucf_test.csv'
        test_dataset = UCFDataset(args.visual_length, test_list, True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        gt = np.load("./data/gt_ucf.npy")
        roc, ap = inference('ucf', model, test_loader, gt, device)
        print(f'roc: {roc}, ap: {ap}')

    elif args.dataset == 'xd':
        test_list = './data/list/xd_test.csv'
        test_dataset = XDDataset(args.visual_length, test_list, True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        gt = np.load("./data/gt_xd.npy")
        roc, ap = inference('xd', model, test_loader, gt, device)
        print(f'roc: {roc}, ap: {ap}')

    elif args.dataset == 'shanghai':
        gt = np.load(r"E:\codes\WSVAD_Datasets\shanghaitech\shanghaitech\testing\weakly setting\gt_shanghai.npy")

        results = []
        for crop_id in range(10):
            print(f"\n{'=' * 60}")
            print(f"开始测试 crop{crop_id}")

            out = test_one_crop_shanghai(args, model, gt, device, crop_id)
            if out is None:
                results.append((crop_id, None, None))
                continue

            roc, ap = out
            results.append((crop_id, roc, ap))
            print(f"crop{crop_id} -> roc: {roc:.6f}, ap: {ap:.6f}")

        print(f"\n{'=' * 60}")
        print("所有 crop 测试结果汇总:")

        valid_results = []
        for crop_id, roc, ap in results:
            if roc is None:
                print(f"crop{crop_id} -> 跳过")
            else:
                print(f"crop{crop_id} -> roc: {roc:.6f}, ap: {ap:.6f}")
                valid_results.append((crop_id, roc, ap))

        if valid_results:
            best_roc = max(valid_results, key=lambda x: x[1])
            best_ap = max(valid_results, key=lambda x: x[2])

            print(f"\n最佳 ROC : crop{best_roc[0]}, roc = {best_roc[1]:.6f}, ap = {best_roc[2]:.6f}")
            print(f"最佳 AP  : crop{best_ap[0]}, roc = {best_ap[1]:.6f}, ap = {best_ap[2]:.6f}")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")