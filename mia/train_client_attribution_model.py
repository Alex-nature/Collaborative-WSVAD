import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ID_COLUMNS = {
    "sample_id",
    "experiment",
    "dataset",
    "split_mode",
    "checkpoint",
    "path",
    "raw_label",
    "source_csv",
    "client_id",
    "sampling_source",
    "origin_sample_id",
    "group_id",
}

LABEL_COLUMN = "client_id"
MEMBERSHIP_COLUMN = "membership"
GROUP_COLUMN = "group_id"


def load_feature_table(csv_path: Path):
    return pd.read_csv(csv_path, low_memory=False)


def select_feature_columns(df: pd.DataFrame):
    feature_columns = []
    for col in df.columns:
        if col in ID_COLUMNS or col == MEMBERSHIP_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")
    return feature_columns


def group_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    groups = df[GROUP_COLUMN].astype(str)
    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(first_split.split(df, groups=groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    trainval_groups = trainval_df[GROUP_COLUMN].astype(str)
    second_split = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(second_split.split(trainval_df, groups=trainval_groups))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def build_lr_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    multi_class="multinomial",
                    class_weight="balanced",
                    random_state=20260319,
                ),
            ),
        ]
    )


def build_rf_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=20260319,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_multiclass(y_true, y_score, labels):
    y_pred = np.argmax(y_score, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    k = min(3, len(labels))
    if k >= 2:
        metrics[f"top{k}_accuracy"] = float(top_k_accuracy_score(y_true, y_score, k=k, labels=np.arange(len(labels))))
    return metrics


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_feature_importance(model_pipeline, feature_columns, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model_pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = np.mean(np.abs(model.coef_), axis=0)
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.full(len(feature_columns), np.nan)
    out_df = pd.DataFrame({"feature": feature_columns, "importance": values})
    out_df = out_df.sort_values("importance", ascending=False, na_position="last")
    out_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=str, help="Path to attack feature CSV")
    parser.add_argument("--name", required=True, type=str, help="Experiment name, e.g. ucf_event_10crop")
    parser.add_argument("--test_size", default=0.2, type=float, help="Test split ratio")
    parser.add_argument("--val_size", default=0.2, type=float, help="Validation split ratio from train split")
    parser.add_argument("--seed", default=20260319, type=int, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    feature_path = project_root / args.features
    df = load_feature_table(feature_path)

    df = df[df[MEMBERSHIP_COLUMN].astype(int) == 1].copy()
    df = df[df[LABEL_COLUMN].astype(str).str.len() > 0].reset_index(drop=True)
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Missing required group column: {GROUP_COLUMN}")

    feature_columns = select_feature_columns(df)
    train_df, val_df, test_df = group_split(df, args.test_size, args.val_size, args.seed)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[LABEL_COLUMN].astype(str))
    y_val = label_encoder.transform(val_df[LABEL_COLUMN].astype(str))
    y_test = label_encoder.transform(test_df[LABEL_COLUMN].astype(str))

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]
    X_test = test_df[feature_columns]

    results = {
        "name": args.name,
        "feature_file": str(feature_path.relative_to(project_root)).replace("\\", "/"),
        "num_rows": int(len(df)),
        "num_features": int(len(feature_columns)),
        "group_column": GROUP_COLUMN,
        "label_column": LABEL_COLUMN,
        "labels": label_encoder.classes_.tolist(),
        "split_sizes": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "split_group_counts": {
            "train": int(train_df[GROUP_COLUMN].nunique()),
            "val": int(val_df[GROUP_COLUMN].nunique()),
            "test": int(test_df[GROUP_COLUMN].nunique()),
        },
        "class_counts": {
            "train": {k: int(v) for k, v in train_df[LABEL_COLUMN].value_counts().sort_index().to_dict().items()},
            "val": {k: int(v) for k, v in val_df[LABEL_COLUMN].value_counts().sort_index().to_dict().items()},
            "test": {k: int(v) for k, v in test_df[LABEL_COLUMN].value_counts().sort_index().to_dict().items()},
        },
    }

    model_specs = {
        "logistic_regression": build_lr_pipeline(),
        "random_forest": build_rf_pipeline(),
    }

    models_dir = project_root / "mia" / "models"
    results_dir = project_root / "mia" / "results"

    for model_name, pipeline in model_specs.items():
        print(f"Training {model_name} for client attribution on {args.name}")
        pipeline.fit(X_train, y_train)

        val_scores = pipeline.predict_proba(X_val)
        test_scores = pipeline.predict_proba(X_test)

        results[model_name] = {
            "validation": evaluate_multiclass(y_val, val_scores, label_encoder.classes_),
            "test": evaluate_multiclass(y_test, test_scores, label_encoder.classes_),
        }

        save_pickle(
            {"pipeline": pipeline, "label_encoder": label_encoder, "feature_columns": feature_columns},
            models_dir / f"{args.name}_{model_name}_client_attribution.pkl",
        )
        save_feature_importance(
            pipeline,
            feature_columns,
            results_dir / f"{args.name}_{model_name}_client_attribution_feature_importance.csv",
        )

    save_json(results, results_dir / f"{args.name}_client_attribution_metrics.json")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
