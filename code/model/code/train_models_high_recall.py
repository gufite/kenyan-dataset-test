from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
RECALL_TARGET = 0.90  # aim to minimize false negatives
ROOT = Path(__file__).resolve().parents[2]
DATASETS: Dict[str, Path] = {
    "full": ROOT / "data/feature-generated/kenya_engineered_features.csv",
    "borrower": ROOT
    / "data/feature-generated/kenya_engineered_features_borrower_side.csv",
}
OUTPUT_DIR = ROOT / "code" / "model" / "outputs_high_recall"
TARGET_COL = "target"
ID_COLS = ["customer_id", "tbl_loan_id"]
GROUP_COL_CANDIDATES = ["customer_id", "customerId", "client_id"]
DATE_COL_CANDIDATES = ["pseudo_disb_date", "disb_date", "disbursement_date", "application_date", "loan_date"]
FEATURES_TO_DROP = {
    "interest_rate",
    "repayment_intensity",
    "lender_risk_profile",
    "pseudo_disb_date",
}


def build_preprocessor(
    feature_frame: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = feature_frame.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in feature_frame.columns if c not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def split_data_leakage_safe(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    date_col = find_first_existing_col(df, DATE_COL_CANDIDATES)
    group_col = find_first_existing_col(df, GROUP_COL_CANDIDATES)

    if date_col is not None:
        tmp = df[[date_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        if tmp[date_col].notna().mean() > 0.8:
            order = tmp[date_col].sort_values().index
            cutoff = int(len(order) * 0.8)
            train_idx = order[:cutoff]
            test_idx = order[cutoff:]
            return (
                X.loc[train_idx],
                X.loc[test_idx],
                y.loc[train_idx],
                y.loc[test_idx],
                f"time_split({date_col})",
            )

    if group_col is not None:
        groups = df[group_col]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            f"group_split({group_col})",
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, "stratified_random_split"


def get_models(scale_pos_weight: float) -> Dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "catboost": CatBoostClassifier(
            iterations=400,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            random_seed=RANDOM_STATE,
        ),
    }


def choose_threshold_for_recall(
    y_true: np.ndarray, y_score: np.ndarray, recall_target: float
) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.concatenate([thresholds, [1.0]])  # align lengths
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    mask = recall >= recall_target
    if mask.any():
        idx = np.argmax(f1 * mask)  # best F1 among those meeting recall target
    else:
        idx = np.argmax(recall)  # fallback to max recall

    return float(thresholds[idx]), {
        "precision_at_threshold": float(precision[idx]),
        "recall_at_threshold": float(recall[idx]),
        "f1_at_threshold": float(f1[idx]),
    }


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, title: str, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_models(dataset_name: str, data_path: Path) -> None:
    print(f"=== Training (recall-oriented) on {dataset_name} dataset ===")
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"target column missing in {data_path}")

    X = df.drop(columns=[TARGET_COL] + ID_COLS, errors="ignore")
    X = X.drop(columns=[c for c in FEATURES_TO_DROP if c in X.columns], errors="ignore")
    y = df[TARGET_COL]

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test, split_tag = split_data_leakage_safe(df, X, y)
    print(f"Split used: {split_tag}")
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    models = get_models(scale_pos_weight)
    ds_out = OUTPUT_DIR / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    report_manifest = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_test)[:, 1]

        threshold, thr_metrics = choose_threshold_for_recall(
            y_test.to_numpy(), probas, RECALL_TARGET
        )
        preds = (probas >= threshold).astype(int)

        metrics = {
            "dataset": dataset_name,
            "split": split_tag,
            "model": model_name,
            "threshold": threshold,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "accuracy": accuracy_score(y_test, preds),
            "auc_roc": roc_auc_score(y_test, probas),
            "auc_pr": average_precision_score(y_test, probas),
            "precision_at_threshold": thr_metrics["precision_at_threshold"],
            "recall_at_threshold": thr_metrics["recall_at_threshold"],
            "f1_at_threshold": thr_metrics["f1_at_threshold"],
        }
        metrics_rows.append(metrics)

        cls_report = classification_report(
            y_test,
            preds,
            target_names=["non_default", "default"],
            digits=3,
            zero_division=0,
        )
        report_path = ds_out / f"classification_report_{model_name}.txt"
        report_path.write_text(
            f"Chosen threshold: {threshold:.4f} (target recall >= {RECALL_TARGET})\n\n"
            + cls_report
        )
        report_manifest[f"classification_report_{model_name}"] = str(report_path)

        roc_path = ds_out / f"roc_{model_name}.png"
        pr_path = ds_out / f"pr_{model_name}.png"
        cm_path = ds_out / f"confusion_matrix_{model_name}.png"

        plot_roc(
            y_test, probas, f"{dataset_name.upper()} - {model_name} ROC (prob)", roc_path
        )
        plot_pr(
            y_test, probas, f"{dataset_name.upper()} - {model_name} PR (prob)", pr_path
        )
        plot_confusion(
            y_test,
            preds,
            f"{dataset_name.upper()} - {model_name} Confusion (thr={threshold:.3f})",
            cm_path,
        )

        report_manifest[f"roc_{model_name}"] = str(roc_path)
        report_manifest[f"pr_{model_name}"] = str(pr_path)
        report_manifest[f"confusion_{model_name}"] = str(cm_path)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["dataset", "recall"], ascending=[True, False]
    )
    metrics_path = ds_out / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics -> {metrics_path}")

    manifest_path = ds_out / "artifacts.json"
    manifest_path.write_text(json.dumps(report_manifest, indent=2))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {name}, missing file: {path}")
            continue
        evaluate_models(name, path)


if __name__ == "__main__":
    main()
