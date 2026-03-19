import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import config
from utils.model import Model
from utils.tools import build_negative_prompts, get_prompt_text


def build_model(args, device):
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
        tca_norm=args.tca_norm,
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_label_map(dataset):
    if dataset == "ucf":
        return {
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
    return {
        "A": "normal",
        "B1": "fighting",
        "B2": "shooting",
        "B4": "riot",
        "B5": "abuse",
        "B6": "car accident",
        "G": "explosion",
    }


def manifest_feature_path(project_root: Path, raw_path: str) -> str:
    return str(project_root / raw_path.replace("/data", "data").lstrip("/"))


def extract_group_and_crop(raw_path: str):
    normalized = raw_path.replace("\\", "/")
    match = re.search(r"__(\d+)\.npy$", normalized)
    if match is None:
        return normalized, -1
    crop_id = int(match.group(1))
    group_id = re.sub(r"__\d+\.npy$", "", normalized)
    return group_id, crop_id


def load_feature_tensor(project_root: Path, raw_path: str, visual_length: int):
    local_path = manifest_feature_path(project_root, raw_path)
    feature = np.load(local_path)
    clip_length = feature.shape[0]
    if clip_length < visual_length:
        padded = np.pad(feature, ((0, visual_length - clip_length), (0, 0)), mode="constant", constant_values=0)
        return torch.tensor(padded).unsqueeze(0), clip_length

    split_num = int(clip_length / visual_length) + 1
    split_feat = None
    for i in range(split_num):
        chunk = feature[i * visual_length:i * visual_length + visual_length, :]
        if i == split_num - 1:
            chunk = np.pad(
                chunk,
                ((0, max(0, visual_length - chunk.shape[0])), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        chunk = chunk.reshape(1, visual_length, feature.shape[1])
        split_feat = chunk if split_feat is None else np.concatenate([split_feat, chunk], axis=0)
    return torch.tensor(split_feat), clip_length


def compute_scores(model, dataset, visual, clip_length, device):
    label_map = get_label_map(dataset)
    prompt_text = get_prompt_text(label_map)
    neg_prompt_text, _ = build_negative_prompts(label_map)
    visual = visual.to(device)

    remaining = clip_length
    max_len = visual.shape[1]
    lengths = torch.zeros(int(clip_length / max_len) + 1, dtype=torch.int32)
    for j in range(int(clip_length / max_len) + 1):
        if j == 0 and remaining < max_len:
            lengths[j] = remaining
        elif j == 0 and remaining > max_len:
            lengths[j] = max_len
            remaining -= max_len
        elif remaining > max_len:
            lengths[j] = max_len
            remaining -= max_len
        else:
            lengths[j] = remaining

    with torch.no_grad():
        outputs = model(visual, prompt_text, lengths, is_training=False, neg_text=neg_prompt_text)
        if isinstance(outputs, tuple):
            logits_pos, logits_neg = outputs
        else:
            logits_pos, logits_neg = outputs, None

    logits_pos = logits_pos.reshape(logits_pos.shape[0] * logits_pos.shape[1], logits_pos.shape[2])
    if logits_neg is not None:
        logits_neg = logits_neg.reshape(logits_neg.shape[0] * logits_neg.shape[1], logits_neg.shape[2])

    pos_logits = logits_pos[:clip_length]
    normal_idx = prompt_text.index("normal")
    probs_pos = pos_logits.softmax(dim=-1)
    score_pos = 1.0 - probs_pos[:, normal_idx]

    if logits_neg is not None:
        neg_logits = logits_neg[:clip_length]
        abnormal_idx = neg_prompt_text.index("abnormal")
        tau = 10.0
        score_neg = torch.sigmoid((neg_logits[:, abnormal_idx] - pos_logits[:, normal_idx]) / tau)
    else:
        score_neg = torch.zeros_like(score_pos)

    alpha = 0.9
    fused_score = alpha * score_pos + (1.0 - alpha) * score_neg
    return (
        score_pos.detach().cpu().numpy().astype(np.float64),
        score_neg.detach().cpu().numpy().astype(np.float64),
        fused_score.detach().cpu().numpy().astype(np.float64),
    )


def safe_entropy(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.min()
    if np.allclose(values.sum(), 0.0):
        return 0.0
    probs = values / values.sum()
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def safe_topk_mean(values, k):
    values = np.asarray(values, dtype=np.float64)
    k = min(k, len(values))
    if k <= 0:
        return 0.0
    return float(np.mean(np.sort(values)[-k:]))


def safe_bottomk_mean(values, k):
    values = np.asarray(values, dtype=np.float64)
    k = min(k, len(values))
    if k <= 0:
        return 0.0
    return float(np.mean(np.sort(values)[:k]))


def peak_count(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return 0
    count = 0
    for idx in range(1, len(values) - 1):
        if values[idx] > values[idx - 1] and values[idx] > values[idx + 1]:
            count += 1
    return count


def branch_features(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    diffs = np.diff(values) if len(values) > 1 else np.array([0.0], dtype=np.float64)
    abs_diffs = np.abs(diffs)
    mean_value = float(np.mean(values))
    top1 = float(np.max(values))
    features = {
        f"{prefix}_mean": mean_value,
        f"{prefix}_max": top1,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_var": float(np.var(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_top1": top1,
        f"{prefix}_top5_mean": safe_topk_mean(values, 5),
        f"{prefix}_top10_mean": safe_topk_mean(values, 10),
        f"{prefix}_bottom5_mean": safe_bottomk_mean(values, 5),
        f"{prefix}_bottom10_mean": safe_bottomk_mean(values, 10),
        f"{prefix}_energy": float(np.sum(values ** 2)),
        f"{prefix}_l1": float(np.sum(np.abs(values))),
        f"{prefix}_entropy": safe_entropy(values),
        f"{prefix}_above_mean_ratio": float(np.mean(values > mean_value)),
        f"{prefix}_above_q75_ratio": float(np.mean(values > np.quantile(values, 0.75))),
        f"{prefix}_peak_count": peak_count(values),
        f"{prefix}_diff_mean": float(np.mean(diffs)),
        f"{prefix}_diff_std": float(np.std(diffs)),
        f"{prefix}_abs_diff_mean": float(np.mean(abs_diffs)),
        f"{prefix}_abs_diff_std": float(np.std(abs_diffs)),
    }

    denom = mean_value if abs(mean_value) > 1e-12 else 1e-12
    features[f"{prefix}_top1_over_mean"] = top1 / denom
    features[f"{prefix}_top5_over_mean"] = features[f"{prefix}_top5_mean"] / denom
    features[f"{prefix}_top10_over_mean"] = features[f"{prefix}_top10_mean"] / denom
    return features


def pairwise_features(score_pos, score_neg, fused_score):
    pos_neg_gap = score_pos - score_neg
    pos_fused_gap = score_pos - fused_score
    neg_fused_gap = score_neg - fused_score
    return {
        "pos_neg_gap_mean": float(np.mean(pos_neg_gap)),
        "pos_neg_gap_max": float(np.max(pos_neg_gap)),
        "pos_neg_gap_min": float(np.min(pos_neg_gap)),
        "pos_neg_gap_std": float(np.std(pos_neg_gap)),
        "pos_fused_gap_mean": float(np.mean(pos_fused_gap)),
        "pos_fused_gap_std": float(np.std(pos_fused_gap)),
        "neg_fused_gap_mean": float(np.mean(neg_fused_gap)),
        "neg_fused_gap_std": float(np.std(neg_fused_gap)),
        "pos_neg_corr": safe_corr(score_pos, score_neg),
        "pos_fused_corr": safe_corr(score_pos, fused_score),
        "neg_fused_corr": safe_corr(score_neg, fused_score),
    }


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_feature_row(project_root: Path, model, row, device, visual_length):
    visual, clip_length = load_feature_tensor(project_root, row["path"], visual_length)
    score_pos, score_neg, fused_score = compute_scores(model, row["dataset"], visual, clip_length, device)
    group_id, crop_id = extract_group_and_crop(row["path"])

    feature_row = {
        "sample_id": row["sample_id"],
        "experiment": row["experiment"],
        "dataset": row["dataset"],
        "split_mode": row["split_mode"],
        "checkpoint": row["checkpoint"],
        "path": row["path"],
        "raw_label": row["raw_label"],
        "membership": row["membership"],
        "source_csv": row["source_csv"],
        "client_id": row["client_id"],
        "sampling_source": row.get("sampling_source", "original"),
        "origin_sample_id": row.get("origin_sample_id", row["sample_id"]),
        "group_id": group_id,
        "crop_id": crop_id,
        "clip_length": clip_length,
    }
    feature_row.update(branch_features(score_pos, "pos"))
    feature_row.update(branch_features(score_neg, "neg"))
    feature_row.update(branch_features(fused_score, "fused"))
    feature_row.update(pairwise_features(score_pos, score_neg, fused_score))
    return feature_row


def read_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_feature_rows(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_extraction(manifest_path: Path, output_path: Path, limit: int | None):
    project_root = Path(__file__).resolve().parents[1]
    manifest_rows = read_manifest(manifest_path)
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    if not manifest_rows:
        raise ValueError(f"No rows found in {manifest_path}")

    args = config.parser.parse_args([])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    current_checkpoint = None
    model = None
    feature_rows = []
    progress = tqdm(manifest_rows, desc=f"Extracting {manifest_path.stem}", unit="sample")

    for idx, row in enumerate(progress, start=1):
        checkpoint_path = project_root / row["checkpoint"]
        if current_checkpoint != checkpoint_path:
            model = build_model(args, device)
            model = load_checkpoint(model, checkpoint_path, device)
            current_checkpoint = checkpoint_path
            progress.set_postfix_str(f"checkpoint={checkpoint_path.name}")

        feature_rows.append(
            extract_feature_row(project_root, model, row, device, args.visual_length)
        )

    write_feature_rows(output_path, feature_rows)
    print(f"Saved features to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str, help="Path to balanced manifest CSV")
    parser.add_argument("--output", required=True, type=str, help="Path to output feature CSV")
    parser.add_argument("--limit", default=None, type=int, help="Optional row limit for smoke tests")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_extraction(project_root / args.manifest, project_root / args.output, args.limit)


if __name__ == "__main__":
    main()
