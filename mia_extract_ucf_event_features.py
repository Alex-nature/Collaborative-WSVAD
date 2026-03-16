import os
import argparse
import numpy as np
import torch

from utils import config
from utils.model import Model
from utils.tools import get_prompt_text, build_negative_prompts, process_split


def build_label_map_ucf():
    return dict(
        {
            "Normal": "normal",
            "Abuse": "abuse",
            "Arrest": "arrest",
            "Arson": "arson",
            "Assault": "assault",
            "Burglary": "burglary",
            "Explosion": "explosion",
            "Fighting": "fighting",
            "RoadAccidents": "roadAccidents",
            "Robbery": "robbery",
            "Shooting": "shooting",
            "Shoplifting": "shoplifting",
            "Stealing": "stealing",
            "Vandalism": "vandalism",
        }
    )


def compute_fused_scores_for_sample(
    model,
    visual_np: np.ndarray,
    prompt_text,
    neg_prompt_text,
    device: torch.device,
    visual_length: int = 256,
    max_len: int = 256,
):
    model.eval()
    split_feat, clip_length = process_split(visual_np, visual_length)

    if split_feat.ndim == 3:
        visual = torch.tensor(split_feat, dtype=torch.float32).to(device)
    elif split_feat.ndim == 2:
        visual = torch.tensor(split_feat, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        raise ValueError(f"Unexpected split_feat ndim: {split_feat.ndim}")
    len_cur = int(clip_length)

    length = clip_length
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
    lengths = lengths.to(torch.int)

    with torch.no_grad():
        outputs = model(
            visual, prompt_text, lengths, is_training=False, neg_text=neg_prompt_text
        )
        if isinstance(outputs, tuple):
            logits2, logits2_neg = outputs
        else:
            logits2, logits2_neg = outputs, None

        logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
        if logits2_neg is not None:
            logits2_neg = logits2_neg.reshape(
                logits2_neg.shape[0] * logits2_neg.shape[1], logits2_neg.shape[2]
            )

        pos_logits = logits2[0:len_cur]
        normal_idx = prompt_text.index("normal")
        probs_pos = pos_logits.softmax(dim=-1)
        score_pos = 1.0 - probs_pos[:, normal_idx]

        if logits2_neg is not None and neg_prompt_text is not None:
            neg_logits = logits2_neg[0:len_cur]
            abnormal_idx = neg_prompt_text.index("abnormal")
            tau = 10.0
            score_neg = torch.sigmoid(
                (neg_logits[:, abnormal_idx] - pos_logits[:, normal_idx]) / tau
            )
        else:
            score_neg = torch.zeros_like(score_pos)

        alpha = 1.0
        fused_score = alpha * score_pos + (1.0 - alpha) * score_neg

    return fused_score.cpu().numpy()


def extract_features_for_npz(
    mia_path: str,
    out_path: str,
    model,
    prompt_text,
    neg_prompt_text,
    visual_length: int,
    device: torch.device,
):
    print(f"\n处理: {mia_path}")
    mia = np.load(mia_path, allow_pickle=True)
    features_arr = mia["features"]
    labels_arr = mia["labels"]
    member_flags = mia["member_flags"]
    client_names = mia["client_names"]
    paths = mia["paths"]

    num_samples = len(features_arr)
    print(f"  样本总数: {num_samples}")

    all_attack_features = []
    last_percent = -1
    print("  开始提取攻击特征...")
    for idx, feat in enumerate(features_arr):
        fused_seq = compute_fused_scores_for_sample(
            model,
            visual_np=feat,
            prompt_text=prompt_text,
            neg_prompt_text=neg_prompt_text,
            device=device,
            visual_length=visual_length,
            max_len=256,
        )
        fused_seq = fused_seq.astype(np.float32)
        L = len(fused_seq)

        max_score = float(np.max(fused_seq))
        mean_score = float(np.mean(fused_seq))
        k = min(16, L)
        topk_mean = float(np.mean(np.sort(fused_seq)[-k:]))

        min_score = float(np.min(fused_seq))
        std_score = float(np.std(fused_seq))
        median_score = float(np.median(fused_seq))
        p25 = float(np.percentile(fused_seq, 25))
        p75 = float(np.percentile(fused_seq, 75))
        range_score = max_score - min_score

        argmax_pos = int(np.argmax(fused_seq))
        argmin_pos = int(np.argmin(fused_seq))
        denom = max(1, L - 1)
        argmax_pos_norm = float(argmax_pos / denom)
        argmin_pos_norm = float(argmin_pos / denom)

        threshold = mean_score + std_score
        high_mask = fused_seq > threshold
        n_peaks = int(high_mask.sum())
        frac_high = float(n_peaks / L)

        half = L // 2
        first_half = fused_seq[:half]
        second_half = fused_seq[half:] if half > 0 else fused_seq
        mean_first_half = float(first_half.mean()) if len(first_half) > 0 else mean_score
        mean_second_half = float(second_half.mean()) if len(second_half) > 0 else mean_score
        delta_mean_2nd_1st = mean_second_half - mean_first_half

        feat_vec = [
            max_score,
            mean_score,
            topk_mean,
            min_score,
            std_score,
            median_score,
            p25,
            p75,
            range_score,
            argmax_pos_norm,
            argmin_pos_norm,
            n_peaks,
            frac_high,
            mean_first_half,
            mean_second_half,
            delta_mean_2nd_1st,
        ]
        all_attack_features.append(feat_vec)

        percent = int((idx + 1) * 100 / num_samples)
        if percent % 5 == 0 and percent != last_percent:
            last_percent = percent
            print(f"    进度: {percent:3d}% ({idx + 1}/{num_samples})", end="\r", flush=True)

    print()
    attack_features = np.asarray(all_attack_features, dtype=np.float32)
    print(f"  攻击特征 shape: {attack_features.shape}")
    print(f"  保存到: {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        attack_features=attack_features,
        member_flags=member_flags,
        labels=labels_arr,
        client_names=client_names,
        paths=paths,
    )


def main():
    parser = argparse.ArgumentParser(
        description="批量对 PAMP-FedVAD/models 里的 mia_ucf_event_data*.npz 提取攻击特征"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./PAMP-FedVAD/models",
        help="包含 mia_ucf_event_data*.npz 的目录",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default=None,
        help="CUDA 设备编号（如 0,1）。为空则自动选择",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型 checkpoint 路径；若不指定，则从 utils.config.parser 中读取 --checkpoint",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.models_dir):
        raise FileNotFoundError(f"目录不存在: {args.models_dir}")

    files = sorted(
        f
        for f in os.listdir(args.models_dir)
        if f.endswith(".npz") and "mia_ucf_event_data" in f
    )
    if not files:
        raise RuntimeError(f"目录中未找到 mia_ucf_event_data*.npz: {args.models_dir}")

    print(f"找到 {len(files)} 个待处理文件:")
    for f in files:
        print("  ", f)

    # 复用 config 参数
    args_cfg, _ = config.parser.parse_known_args()
    device = (
        torch.device(f"cuda:{args.cuda}")
        if args.cuda is not None and torch.cuda.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = args.checkpoint if args.checkpoint is not None else args_cfg.checkpoint
    if ckpt_path is None:
        raise ValueError(
            "未指定模型 checkpoint。\n"
            "请在命令行添加参数，例如：\n"
            "  python mia_extract_ucf_event_features_batch.py ... --checkpoint save/xxx/model_final_*.pth"
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"指定的 checkpoint 文件不存在: {ckpt_path}")

    print("构建模型并加载 checkpoint...")
    model = Model(
        args_cfg.embed_dim,
        args_cfg.visual_length,
        args_cfg.prompt_prefix,
        args_cfg.prompt_postfix,
        args_cfg.visual_width,
        args_cfg.visual_head,
        args_cfg.visual_layers,
        str(device),
        use_tca=args_cfg.use_tca,
        tca_window_size=args_cfg.tca_window_size,
        tca_dropout=args_cfg.tca_dropout,
        use_distance_adj=args_cfg.use_distance_adj,
        tca_gamma=args_cfg.tca_gamma,
        tca_bias=args_cfg.tca_bias,
        tca_norm=args_cfg.tca_norm,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    label_map = build_label_map_ucf()
    prompt_text = get_prompt_text(label_map)
    neg_prompt_text, _ = build_negative_prompts(label_map)

    for i, fname in enumerate(files, start=1):
        mia_path = os.path.join(args.models_dir, fname)
        out_name = fname.replace("mia_ucf_event_data", "mia_ucf_event_features")
        out_path = os.path.join(args.models_dir, out_name)
        print(f"\n=== [{i}/{len(files)}] ===")
        extract_features_for_npz(
            mia_path,
            out_path,
            model,
            prompt_text,
            neg_prompt_text,
            visual_length=args_cfg.visual_length,
            device=device,
        )

    print("\n全部完成。")


if __name__ == "__main__":
    main()