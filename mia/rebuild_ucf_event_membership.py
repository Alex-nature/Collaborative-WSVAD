import argparse
import csv
from pathlib import Path


DATASET_CONFIG = {
    "ucf": {
        "experiment": "ucf_event_aug",
        "member_dir": "data/list/ucf_event",
        "test_csv": "data/list/ucf_test.csv",
        "member_csv": "mia/manifests/ucf_event_members.csv",
        "nonmember_csv": "mia/manifests/ucf_event_nonmembers_10crop.csv",
        "combined_csv": "mia/manifests/membership_manifest_ucf_event_10crop.csv",
        "default_checkpoint": "PAMP-FedVAD/models/ucf-event_model_roc_0.8797.pth",
    },
    "xd": {
        "experiment": "xd_event_aug",
        "member_dir": "data/list/xd_event",
        "test_csv": "data/list/xd_test.csv",
        "member_csv": "mia/manifests/xd_event_members.csv",
        "nonmember_csv": "mia/manifests/xd_event_nonmembers_10crop.csv",
        "combined_csv": "mia/manifests/membership_manifest_xd_event_10crop.csv",
        "default_checkpoint": "PAMP-FedVAD/models/xd-event_model_final_ap_0.8243.pth",
    },
}


def normalize_path(raw_path: str) -> str:
    return raw_path.replace("\\", "/")


def read_csv_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(csv_path: Path, fieldnames, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_member_rows(project_root: Path, member_dir: str):
    member_rows = []
    source_dir = project_root / member_dir
    for csv_path in sorted(source_dir.glob("*.csv")):
        client_id = csv_path.stem
        for row in read_csv_rows(csv_path):
            member_rows.append(
                {
                    "path": normalize_path(row["path"]),
                    "raw_label": row["label"],
                    "source_csv": normalize_path(str(csv_path.relative_to(project_root))),
                    "client_id": client_id,
                }
            )
    return member_rows


def expand_crop_paths(project_root: Path, crop5_path: str):
    expanded = []
    base_rel = crop5_path.replace("/data", "data").lstrip("/")
    base_path = project_root / base_rel

    if "__5.npy" in base_path.name:
        base_name = "__5.npy"
    elif "__0.npy" in base_path.name:
        base_name = "__0.npy"
    else:
        return expanded

    for crop_id in range(10):
        crop_path = Path(str(base_path).replace(base_name, f"__{crop_id}.npy"))
        if crop_path.exists():
            rel_path = normalize_path(str(crop_path.relative_to(project_root)))
            expanded.append("/" + rel_path)
    return expanded


def build_nonmember_rows(project_root: Path, test_csv: str):
    source_csv = project_root / test_csv
    nonmember_rows = []
    for row in read_csv_rows(source_csv):
        expanded_paths = expand_crop_paths(project_root, row["path"])
        for expanded_path in expanded_paths:
            nonmember_rows.append(
                {
                    "path": expanded_path,
                    "raw_label": row["label"],
                    "source_csv": normalize_path(str(source_csv.relative_to(project_root))),
                    "crop_source_path": normalize_path(row["path"]),
                }
            )
    return nonmember_rows


def build_combined_manifest(experiment: str, dataset: str, checkpoint: str, member_rows, nonmember_rows):
    combined_rows = []
    for idx, row in enumerate(member_rows):
        combined_rows.append(
            {
                "sample_id": f"{experiment}::{idx:07d}",
                "experiment": experiment,
                "dataset": dataset,
                "split_mode": "event",
                "checkpoint": checkpoint,
                "path": row["path"],
                "raw_label": row["raw_label"],
                "membership": 1,
                "source_csv": row["source_csv"],
                "client_id": row["client_id"],
            }
        )

    base_idx = len(combined_rows)
    for offset, row in enumerate(nonmember_rows):
        combined_rows.append(
            {
                "sample_id": f"{experiment}::{base_idx + offset:07d}",
                "experiment": experiment,
                "dataset": dataset,
                "split_mode": "event",
                "checkpoint": checkpoint,
                "path": row["path"],
                "raw_label": row["raw_label"],
                "membership": 0,
                "source_csv": row["source_csv"],
                "client_id": "",
            }
        )
    return combined_rows


def run_for_dataset(project_root: Path, dataset: str, checkpoint: str | None):
    cfg = DATASET_CONFIG[dataset]
    checkpoint = checkpoint or cfg["default_checkpoint"]

    member_rows = build_member_rows(project_root, cfg["member_dir"])
    nonmember_rows = build_nonmember_rows(project_root, cfg["test_csv"])
    combined_rows = build_combined_manifest(
        cfg["experiment"], dataset, checkpoint, member_rows, nonmember_rows
    )

    write_csv_rows(
        project_root / cfg["member_csv"],
        ["path", "raw_label", "source_csv", "client_id"],
        member_rows,
    )
    write_csv_rows(
        project_root / cfg["nonmember_csv"],
        ["path", "raw_label", "source_csv", "crop_source_path"],
        nonmember_rows,
    )
    write_csv_rows(
        project_root / cfg["combined_csv"],
        [
            "sample_id",
            "experiment",
            "dataset",
            "split_mode",
            "checkpoint",
            "path",
            "raw_label",
            "membership",
            "source_csv",
            "client_id",
        ],
        combined_rows,
    )

    print(
        {
            "dataset": dataset,
            "members": len(member_rows),
            "nonmembers_10crop": len(nonmember_rows),
            "combined": len(combined_rows),
            "member_csv": cfg["member_csv"],
            "nonmember_csv": cfg["nonmember_csv"],
            "combined_manifest": cfg["combined_csv"],
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ucf", "xd", "all"], default="ucf", help="Dataset to rebuild")
    parser.add_argument("--checkpoint", default=None, type=str, help="Optional checkpoint override")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    datasets = ["ucf", "xd"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_for_dataset(project_root, dataset, args.checkpoint)


if __name__ == "__main__":
    main()
