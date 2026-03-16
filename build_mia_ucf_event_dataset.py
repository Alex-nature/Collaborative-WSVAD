import os
import argparse
from typing import Tuple, List
import numpy as np
import pandas as pd
import time  # 用于可选的动态随机种子


def load_ucf_event_clients(
    event_dir: str,
    base_data_root: str = "./data",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    从 ucf_event 目录中加载每个客户端（事件）的 CSV，
    返回合并后的 DataFrame 及每条样本对应的 client_name。

    client_name 直接用文件名中的事件名，例如:
        ./data/list/ucf_event/ucf_Abuse.csv -> client_name = "Abuse"
    """
    if not os.path.isdir(event_dir):
        raise FileNotFoundError(f"UCF event dir not found: {event_dir}")

    all_rows = []
    client_names = []

    for fname in sorted(os.listdir(event_dir)):
        if not fname.startswith("ucf_") or not fname.endswith(".csv"):
            continue
        csv_path = os.path.join(event_dir, fname)
        df = pd.read_csv(csv_path)
        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError(f"CSV must contain 'path' and 'label' columns: {csv_path}")

        # 解析客户端名（事件名）
        # 例如 ucf_Abuse.csv -> Abuse
        base = os.path.splitext(fname)[0]
        client_name = base.replace("ucf_", "")

        # 统一路径前缀
        df = df.copy()
        df["feature_path"] = df["path"].astype(str).map(
            lambda p: p.replace("/data", base_data_root)
        )
        df["client_name"] = client_name
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"No valid ucf_*.csv found in {event_dir}")

    merged = pd.concat(all_rows, axis=0, ignore_index=True)
    return merged, merged["client_name"].tolist()


def load_ucf_test(
    test_csv: str,
    base_data_root: str = "./data",
) -> pd.DataFrame:
    """
    加载 UCF 测试划分（作为非成员样本来源），统一路径前缀。
    """
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df_test = pd.read_csv(test_csv)
    if "path" not in df_test.columns or "label" not in df_test.columns:
        raise ValueError(f"Test CSV must contain 'path' and 'label' columns: {test_csv}")

    df_test = df_test.copy()
    df_test["feature_path"] = df_test["path"].astype(str).map(
        lambda p: p.replace("/data", base_data_root)
    )
    # 测试集不属于任何客户端，这里用特殊 client_name 标记
    df_test["client_name"] = "TEST"
    return df_test


def sample_member_nonmember_event(
    df_train_all: pd.DataFrame,
    df_test: pd.DataFrame,
    num_member: int = None,
    num_nonmember: int = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    在 event 场景下采样成员 / 非成员：
      - 成员：来自所有客户端训练 CSV 的并集（df_train_all）
      - 非成员：来自测试集 df_test
    """
    df_train_unique = df_train_all.drop_duplicates(subset=["feature_path"]).reset_index(drop=True)
    df_test_unique = df_test.drop_duplicates(subset=["feature_path"]).reset_index(drop=True)

    max_member = len(df_train_unique)
    max_nonmember = len(df_test_unique)

    # 全量优先：
    # - 若未显式指定 num_member / num_nonmember，则默认使用各自全部样本
    # - 若指定了数量，则在不超过各自最大样本数的前提下采样对应数量
    if num_member is None:
        num_member = max_member
    if num_nonmember is None:
        num_nonmember = max_nonmember

    num_member = min(num_member, max_member)
    num_nonmember = min(num_nonmember, max_nonmember)

    df_member = df_train_unique.sample(
        n=num_member, random_state=random_state, replace=False
    ).reset_index(drop=True)
    df_nonmember = df_test_unique.sample(
        n=num_nonmember, random_state=random_state + 1, replace=False
    ).reset_index(drop=True)

    return df_member, df_nonmember


def load_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据 feature_path 加载 .npy 特征，同时返回标签、原始路径和 client_name。
    """
    features = []
    labels = []
    paths = []
    client_names = []

    for _, row in df.iterrows():
        fpath = str(row["feature_path"])
        label = row["label"]
        cname = row["client_name"]

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Feature file not found: {fpath}")

        feat = np.load(fpath)

        features.append(feat)
        labels.append(label)
        paths.append(fpath)
        client_names.append(cname)

    features_arr = np.array(features, dtype=object)
    labels_arr = np.array(labels)
    paths_arr = np.array(paths)
    client_arr = np.array(client_names)

    return features_arr, labels_arr, paths_arr, client_arr


def main():
    parser = argparse.ArgumentParser(
        description="构造 UCF-event 场景下的成员推断数据集（成员=各客户端训练集，非成员=测试集，记录 client_name）"
    )
    parser.add_argument(
        "--event_dir",
        type=str,
        default="./data/list/ucf_event",
        help="UCF event 划分目录（每个客户端一个 ucf_*.csv）",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="./data/list/ucf_test.csv",
        help="UCF 测试集列表 CSV（非成员样本来源）",
    )
    parser.add_argument(
        "--base_data_root",
        type=str,
        default="./data",
        help="替换路径中 /data 前缀的本地数据根目录",
    )
    parser.add_argument(
        "--num_member",
        type=int,
        default=290,
        help="成员样本数量（默认与非成员平衡，取 train/test 中可用数量的最小值）",
    )
    parser.add_argument(
        "--num_nonmember",
        type=int,
        default=290,
        help="非成员样本数量（默认与成员平衡，取 train/test 中可用数量的最小值）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./PAMP-FedVAD/models/mia_ucf_event_data.npz",
        help="输出 npz 文件路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（不填则用当前时间戳）；用于采样成员 / 非成员",
    )
    parser.add_argument(
        "--num_datasets",
        type=int,
        default=20,
        help="要构建的 npz 数据集数量（例如 20）。>1 时会生成多个文件",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=1,
        help="当 num_datasets>1 时，每个数据集 seed 的步长（seed_i = seed + i*seed_step）",
    )

    args = parser.parse_args()

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.splitext(os.path.basename(args.output))[0]

    # 使用当前时间戳生成基础随机种子
    seed_base = int(time.time()) if args.seed is None else int(args.seed)

    print("加载 UCF event 训练客户端划分...")
    df_train_all, _ = load_ucf_event_clients(
        event_dir=args.event_dir,
        base_data_root=args.base_data_root,
    )
    print(f"  训练样本数（所有客户端并集）: {len(df_train_all)}")

    print("加载 UCF 测试集...")
    df_test = load_ucf_test(
        test_csv=args.test_csv,
        base_data_root=args.base_data_root,
    )
    print(f"  测试样本数: {len(df_test)}")

    # 非成员样本默认全量（或按 num_nonmember 指定），可复用以避免重复 I/O
    print("准备非成员样本（可复用）...")
    _, df_nonmember_base = sample_member_nonmember_event(
        df_train_all,
        df_test,
        num_member=0,
        num_nonmember=args.num_nonmember,
        random_state=seed_base + 999,
    )
    print(f"  非成员样本数: {len(df_nonmember_base)}")
    print("  加载非成员样本特征...")
    nonmember_features, nonmember_labels, nonmember_paths, nonmember_clients = load_features_and_labels(
        df_nonmember_base
    )

    for i in range(int(args.num_datasets)):
        # 在循环内使用当前时间戳作为随机种子
        seed_i = int(time.time())  # 每次采样时使用当前时间戳
        out_path = (
            args.output
            if int(args.num_datasets) == 1
            else os.path.join(out_dir, f"{out_base}_seed{seed_i}.npz")
        )

        print(f"\n=== 构建数据集 {i + 1}/{args.num_datasets} (seed={seed_i}) ===")

        print("采样成员样本...")
        df_member, _ = sample_member_nonmember_event(
            df_train_all,
            df_test,
            num_member=args.num_member,
            num_nonmember=0,
            random_state=seed_i,  # 使用当前时间戳作为种子
        )
        print(f"  成员样本数: {len(df_member)}")

        print("加载成员样本特征...")
        member_features, member_labels, member_paths, member_clients = load_features_and_labels(df_member)

        print("合并并构建标签...")
        features = np.concatenate([member_features, nonmember_features], axis=0)
        labels = np.concatenate([member_labels, nonmember_labels], axis=0)
        paths = np.concatenate([member_paths, nonmember_paths], axis=0)
        client_names = np.concatenate([member_clients, nonmember_clients], axis=0)

        member_flags = np.concatenate(
            [
                np.ones(len(member_features), dtype=np.int64),
                np.zeros(len(nonmember_features), dtype=np.int64),
            ],
            axis=0,
        )

        print(
            f"最终样本数: {len(features)} "
            f"(成员 {len(member_features)} / 非成员 {len(nonmember_features)})"
        )
        print(f"保存到: {out_path}")

        np.savez_compressed(
            out_path,
            features=features,
            labels=labels,
            member_flags=member_flags,
            paths=paths,
            client_names=client_names,
            seed=np.int64(seed_i),
        )

    print("\n完成。")


if __name__ == "__main__":
    main()