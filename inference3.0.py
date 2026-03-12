import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from utils import config
from utils.dataset import XDDataset, UCFDataset
from utils.model import Model
from utils.tools import get_prompt_text, build_negative_prompts


def plot_score(scores, gt_values, save_path, video_name, stride=16,
               figsize=(12, 1.8), line_width=1.2, fill_alpha=0.35,
               show_title=False, show_legend=False):
    scores = np.asarray(scores).reshape(-1)
    gt_values = np.asarray(gt_values).reshape(-1)

    # 1) 片段级 -> 帧级
    scores_frame = np.repeat(scores, stride)

    # 2) 对齐长度
    L = min(len(scores_frame), len(gt_values))
    scores_frame = scores_frame[:L]
    gt_values = gt_values[:L]

    x = np.arange(L)

    fig, ax = plt.subplots(figsize=figsize)

    # 曲线
    ax.plot(x, scores_frame, linewidth=line_width)

    # GT 区间填充
    gt_bin = (gt_values > 0.5).astype(np.int32)
    start_idx = None
    for i, v in enumerate(gt_bin):
        if v == 1 and start_idx is None:
            start_idx = i
        elif v == 0 and start_idx is not None:
            ax.axvspan(start_idx, i - 1, facecolor='lightcoral', alpha=fill_alpha, edgecolor='none')
            start_idx = None
    if start_idx is not None:
        ax.axvspan(start_idx, i - 1, facecolor='lightcoral', alpha=fill_alpha, edgecolor='none')

    # 轴范围
    left_margin_ratio = 0.025
    left_margin = max(1, int(L * left_margin_ratio))
    ax.set_xlim(-left_margin, L - 1 if L > 0 else 1)
    ax.set_ylim(0, 1)

    # ✅ 关键：y轴刻度固定为 0 / 0.5 / 1
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'])


    # 论文风格
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8, length=2)

    # 放大并加粗横纵坐标刻度字体
    ax.tick_params(axis='both', labelsize=12, length=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    
    ax.set_xlabel('')
    ax.set_ylabel('')

    if show_title:
        ax.set_title(f'{os.path.basename(video_name)}', fontsize=10, pad=2)

    if show_legend:
        from matplotlib.patches import Patch
        legend_elements = [
            plt.Line2D([0], [0], label='Score'),
            Patch(alpha=fill_alpha, label='GT (Anomaly)')
        ]
        ax.legend(handles=legend_elements, fontsize=8, frameon=False, loc='upper right')

    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.28)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def inference(dataset, model, test_loader, gt, device):
    visual_length = 256
    stride = 16  # 每个特征对应多少帧（与你的np.repeat保持一致）

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

    # ===== 可视化保存目录 =====
    save_dir = 'PAMP-FedVAD/results/anomaly_scores'
    os.makedirs(save_dir, exist_ok=True)

    start_frame = 0  # 全局gt切片起点（帧级）

    with torch.no_grad():
        max_len = 256
        ap2 = None

        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0).to(device)
            length = int(item[2])
            len_cur = length

            if len_cur < visual_length:
                visual = visual.unsqueeze(0)

            # ===== lengths 切块逻辑 =====
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

            # ===== 前向（带负prompt）=====
            outputs = model(visual, prompt_text, lengths, is_training=False, neg_text=neg_prompt_text)
            if isinstance(outputs, tuple):
                logits2, logits2_neg = outputs
            else:
                logits2, logits2_neg = outputs, None

            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            if logits2_neg is not None:
                logits2_neg = logits2_neg.reshape(logits2_neg.shape[0] * logits2_neg.shape[1], logits2_neg.shape[2])

            # ===== 分数计算 =====
            pos_logits = logits2[0:len_cur]                 # [L, C_pos]
            normal_idx = prompt_text.index('normal')

            probs_pos = pos_logits.softmax(dim=-1)
            score_pos = 1.0 - probs_pos[:, normal_idx]      # [L]

            if logits2_neg is not None:
                neg_logits = logits2_neg[0:len_cur]          # [L, C_neg]
                abnormal_idx = neg_prompt_text.index('abnormal')

                tau = 10.0
                score_neg = torch.sigmoid(
                    (neg_logits[:, abnormal_idx] - pos_logits[:, normal_idx]) / tau
                )                                           # [L]
            else:
                score_neg = torch.zeros_like(score_pos)

            alpha = 1.0
            fused_score = alpha * score_pos + (1.0 - alpha) * score_neg  # [L]
            prob2 = fused_score.reshape(-1)  # [L]

            # ====== 帧级GT切片 + 保存曲线图 ======
            end_frame = start_frame + (len_cur * stride)
            gt_segment = gt[start_frame:end_frame]

            # 视频名/类别（更通用一点，不强依赖 __5.npy）
            feature_path = test_loader.dataset.df.iloc[i]['path']
            category = os.path.basename(os.path.dirname(feature_path))
            base = os.path.splitext(os.path.basename(feature_path))[0]
            video_name = base.replace('__5', '')

            category_dir = os.path.join(save_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            save_path = os.path.join(category_dir, f'{video_name}_scores.png')

            # 生成“扁扁的”曲线图：高度小
            plot_score(prob2.cpu().numpy(), gt_segment, save_path, video_name,
                       stride=stride, figsize=(12, 1.8),
                       line_width=1.2, fill_alpha=0.35,
                       show_title=False, show_legend=False)

            start_frame = end_frame

            # ===== 汇总用于指标 =====
            ap2 = prob2 if ap2 is None else torch.cat([ap2, prob2], dim=0)

    ap2 = ap2.detach().cpu().numpy().tolist()

    # 帧级预测（repeat到每帧）
    pred_frame = np.repeat(ap2, stride)

    # 对齐长度
    L = min(len(gt), len(pred_frame))
    ROC2 = roc_auc_score(gt[:L], pred_frame[:L])
    AP2 = average_precision_score(gt[:L], pred_frame[:L])

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
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if args.dataset == 'ucf':
        gt = np.load("./data/gt_ucf.npy")
        roc, ap = inference('ucf', model, test_loader, gt, device)
    else:
        gt = np.load("./data/gt_xd.npy")
        roc, ap = inference('xd', model, test_loader, gt, device)

    print(f'roc: {roc}, ap: {ap}')