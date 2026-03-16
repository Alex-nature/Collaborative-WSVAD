import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def run_single_npz(
    features_npz: str,
    test_size: float,
    random_state: int,
    use_only_basic3: bool,
):
    print(f"加载特征文件: {features_npz}")
    data = np.load(features_npz, allow_pickle=True)

    X = data["attack_features"]  # [N, D]
    y = data["member_flags"].astype(int)  # [N]

    print(f"attack_features shape: {X.shape}, member_flags shape: {y.shape}")

    if use_only_basic3:
        if X.shape[1] < 3:
            raise ValueError("attack_features 维度 < 3，无法只用 basic3 特征")
        print("仅使用前三个基本特征：[max_score, mean_score, topk_mean]")
        X = X[:, :3]

    # 简单随机划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(
        f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本, 特征维度: {X_train.shape[1]}"
    )

    # 使用标准化 + 逻辑回归的简单 pipeline
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=200,
            n_jobs=-1,
        ),
    )

    print("开始训练攻击模型（Logistic Regression）...")
    clf.fit(X_train, y_train)

    # 预测概率（成员 = 正类 1）
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    # 计算指标
    roc = roc_auc_score(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    pos_ratio = float(y.mean())

    return {
        "roc_auc": float(roc),
        "ap": float(ap),
        "acc": float(acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pos_ratio": pos_ratio,
        "n": int(len(y)),
        "d": int(X.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Baseline 成员推断攻击：基于 mia_ucf_event_features.npz 训练一个简单攻击器"
    )
    parser.add_argument(
        "--features_npz",
        type=str,
        default="./PAMP-FedVAD/models/mia_ucf_event_features.npz",
        help="由 mia_extract_ucf_event_features.py 生成的特征 npz 路径",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="批量模式：包含多个 mia_ucf_event_features*.npz 的目录（指定后会忽略 --features_npz）",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="mia_ucf_event_features*.npz",
        help="批量模式：在 --features_dir 下匹配的文件名模式（默认: mia_ucf_event_features*.npz）",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="测试集占比（0~1）",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--use_only_basic3",
        action="store_true",
        help="若指定，则只使用前三个基本特征 [max, mean, topk_mean]",
    )

    args = parser.parse_args()

    # 单文件模式
    if args.features_dir is None:
        metrics = run_single_npz(
            args.features_npz,
            test_size=args.test_size,
            random_state=args.random_state,
            use_only_basic3=args.use_only_basic3,
        )
        print("====== Baseline MIA 结果 ======")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"AP     : {metrics['ap']:.4f}")
        print(f"ACC    : {metrics['acc']:.4f}")
        print(
            f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}"
        )
        print("================================")
        print(f"整体成员比例 (positive rate): {metrics['pos_ratio']:.4f}")
        return

    # 批量模式
    features_dir = args.features_dir
    if not os.path.isdir(features_dir):
        raise FileNotFoundError(f"--features_dir 不是有效目录: {features_dir}")

    import fnmatch

    files = sorted(
        f
        for f in os.listdir(features_dir)
        if f.endswith(".npz") and fnmatch.fnmatch(f, args.pattern)
    )
    if not files:
        raise RuntimeError(f"目录下未找到匹配文件: {features_dir} / pattern={args.pattern}")

    print(f"批量模式：找到 {len(files)} 个特征文件")

    all_metrics = []
    for i, fname in enumerate(files, start=1):
        fpath = os.path.join(features_dir, fname)
        print(f"\n=== [{i}/{len(files)}] {fname} ===")
        metrics = run_single_npz(
            fpath,
            test_size=args.test_size,
            random_state=args.random_state,
            use_only_basic3=args.use_only_basic3,
        )
        all_metrics.append(metrics)
        print("结果：")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  AP     : {metrics['ap']:.4f}")
        print(f"  ACC    : {metrics['acc']:.4f}")
        print(f"  N={metrics['n']}, D={metrics['d']}, pos_rate={metrics['pos_ratio']:.4f}")

    rocs = np.array([m["roc_auc"] for m in all_metrics], dtype=np.float64)
    aps = np.array([m["ap"] for m in all_metrics], dtype=np.float64)
    accs = np.array([m["acc"] for m in all_metrics], dtype=np.float64)

    print("\n====== 批量 Baseline MIA 汇总（均值 ± 标准差）======")
    print(f"ROC-AUC: {rocs.mean():.4f} ± {rocs.std(ddof=1):.4f}")
    print(f"AP     : {aps.mean():.4f} ± {aps.std(ddof=1):.4f}")
    print(f"ACC    : {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
    print("===================================================")


if __name__ == "__main__":
    main()