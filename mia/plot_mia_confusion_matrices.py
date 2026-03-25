import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mia.train_attack_model import GROUP_COLUMN, LABEL_COLUMN, group_split, load_feature_table, select_feature_columns


def load_model(model_path: Path):
    with model_path.open("rb") as f:
        return pickle.load(f)


def plot_confusion(y_true, y_pred, title: str, output_path: Path, normalize: str | None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize=normalize)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-member", "member"])
    value_format = ".2f" if normalize else "d"
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=value_format)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=str, help="Path to attack feature CSV")
    parser.add_argument("--name", required=True, type=str, help="Experiment name, e.g. ucf_event_10crop")
    parser.add_argument("--test_size", default=0.2, type=float, help="Test split ratio")
    parser.add_argument("--val_size", default=0.2, type=float, help="Validation split ratio from train split")
    parser.add_argument("--seed", default=20260319, type=int, help="Random seed")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Plot normalized confusion matrices instead of raw counts",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    feature_path = project_root / args.features
    df = load_feature_table(feature_path)
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Missing required group column: {GROUP_COLUMN}")

    feature_columns = select_feature_columns(df)
    _, _, test_df = group_split(df, args.test_size, args.val_size, args.seed)

    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN].astype(int).to_numpy()

    models_dir = project_root / "mia" / "models"
    results_dir = project_root / "mia" / "results"
    model_specs = {
        "logistic_regression": models_dir / f"{args.name}_logistic_regression.pkl",
        "random_forest": models_dir / f"{args.name}_random_forest.pkl",
    }
    normalize_mode = "true" if args.normalize else None

    for model_name, model_path in model_specs.items():
        pipeline = load_model(model_path)
        y_pred = pipeline.predict(X_test)
        suffix = "normalized" if args.normalize else "raw"
        output_path = results_dir / f"{args.name}_{model_name}_confusion_matrix_{suffix}.png"
        plot_confusion(
            y_test,
            y_pred,
            title=f"{args.name} - {model_name.replace('_', ' ').title()}",
            output_path=output_path,
            normalize=normalize_mode,
        )
        print(f"Saved confusion matrix: {output_path}")


if __name__ == "__main__":
    main()
