import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
}

LABEL_COLUMN = "membership"


def load_feature_table(csv_path: Path):
    df = pd.read_csv(csv_path)
    return df


def select_feature_columns(df: pd.DataFrame):
    feature_columns = []
    for col in df.columns:
        if col == LABEL_COLUMN or col in ID_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")
    return feature_columns


def compute_tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def evaluate_predictions(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tpr_at_fpr_1pct": compute_tpr_at_fpr(y_true, y_score, 0.01),
        "tpr_at_fpr_5pct": compute_tpr_at_fpr(y_true, y_score, 0.05),
    }


def build_lr_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
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
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=20260319,
                    n_jobs=-1,
                ),
            ),
        ]
    )


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
        values = np.abs(model.coef_).reshape(-1)
        df = pd.DataFrame({"feature": feature_columns, "importance": values})
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        df = pd.DataFrame({"feature": feature_columns, "importance": values})
    else:
        df = pd.DataFrame({"feature": feature_columns, "importance": np.nan})

    df = df.sort_values("importance", ascending=False, na_position="last")
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=str, help="Path to attack feature CSV")
    parser.add_argument("--name", required=True, type=str, help="Experiment name, e.g. ucf_event")
    parser.add_argument("--test_size", default=0.2, type=float, help="Test split ratio")
    parser.add_argument("--val_size", default=0.2, type=float, help="Validation split ratio from train split")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    feature_path = project_root / args.features
    df = load_feature_table(feature_path)

    feature_columns = select_feature_columns(df)
    X = df[feature_columns]
    y = df[LABEL_COLUMN].astype(int).to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=20260319
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_size, stratify=y_trainval, random_state=20260319
    )

    results = {
        "name": args.name,
        "feature_file": str(feature_path.relative_to(project_root)).replace("\\", "/"),
        "num_rows": int(len(df)),
        "num_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "split_sizes": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
    }

    model_specs = {
        "logistic_regression": build_lr_pipeline(),
        "random_forest": build_rf_pipeline(),
    }

    models_dir = project_root / "mia" / "models"
    results_dir = project_root / "mia" / "results"

    for model_name, pipeline in model_specs.items():
        print(f"Training {model_name} on {args.name}")
        pipeline.fit(X_train, y_train)

        val_scores = pipeline.predict_proba(X_val)[:, 1]
        test_scores = pipeline.predict_proba(X_test)[:, 1]

        results[model_name] = {
            "validation": evaluate_predictions(y_val, val_scores),
            "test": evaluate_predictions(y_test, test_scores),
        }

        save_pickle(pipeline, models_dir / f"{args.name}_{model_name}.pkl")
        save_feature_importance(
            pipeline,
            feature_columns,
            results_dir / f"{args.name}_{model_name}_feature_importance.csv",
        )

    save_json(results, results_dir / f"{args.name}_metrics.json")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
