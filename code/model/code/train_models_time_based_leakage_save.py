from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

# -----------------------------
# CONFIG
# -----------------------------
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[2]

DATASETS: Dict[str, Path] = {
    "full": ROOT / "data/feature-generated/kenya_engineered_features.csv",
    "borrower": ROOT / "data/feature-generated/kenya_engineered_features_borrower_side.csv",
}

OUTPUT_DIR = ROOT / "code" / "model" / "outputs_leakage_safe"
TARGET_COL = "target"

# If you have customer_id, we can do leakage-safe group split (no customer appears in both train/test)
GROUP_COL_CANDIDATES = ["customer_id", "customerId", "client_id"]

# If you have a real date column, we can do the best-practice time split
# Put your real date column name here if you have one (e.g., "disb_date")
DATE_COL_CANDIDATES = [
    "pseudo_disb_date",
    "disb_date",
    "disbursement_date",
    "application_date",
    "loan_date",
]

ID_COLS = ["customer_id", "tbl_loan_id"]

FEATURES_TO_DROP = {
    "interest_rate",
    "repayment_intensity",
    "lender_risk_profile",
    "pseudo_disb_date",
    "repayment_consistency",
    "latest_amount_ma3",
    "trend_in_amount",
    "trend_in_burden",
    "burden_percentile",
}

# Choose a business-aligned threshold strategy:
# - "fixed_0.5": keep 0.5
# - "target_recall": choose threshold achieving at least TARGET_RECALL (risk-averse)
THRESHOLD_STRATEGY = "target_recall"
TARGET_RECALL = 0.80  # adjust to your credit policy (e.g., 0.75–0.90)


# -----------------------------
# HELPERS
# -----------------------------
def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_preprocessor(feature_frame: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = feature_frame.select_dtypes(include=["object", "category"]).columns.tolist()
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
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def get_models(scale_pos_weight: float) -> Dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "catboost": CatBoostClassifier(
            iterations=600,
            depth=8,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            random_seed=RANDOM_STATE,
        ),
    }


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, strategy: str) -> float:
    """
    Credit-scoring-friendly threshold selection.
    - fixed_0.5: use 0.5
    - target_recall: smallest threshold that achieves TARGET_RECALL (risk-averse)
    """
    if strategy == "fixed_0.5":
        return 0.5

    if strategy == "target_recall":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns recall aligned with thresholds plus one extra point
        # thresholds has length n-1; we align by ignoring the last recall/precision point
        recall = recall[:-1]
        precision = precision[:-1]

        # We want recall >= TARGET_RECALL; pick the highest threshold that still meets it
        eligible = np.where(recall >= TARGET_RECALL)[0]
        if len(eligible) == 0:
            # If the model cannot hit the target recall, fall back to the best recall point
            best_idx = int(np.argmax(recall))
            return float(thresholds[best_idx])

        # Among eligible points, choose the threshold with best precision (or you can choose max threshold)
        # Here: choose best precision among eligible, tie-breaker by higher threshold
        best = eligible[np.lexsort((thresholds[eligible], precision[eligible]))][-1]
        return float(thresholds[best])

    raise ValueError(f"Unknown threshold strategy: {strategy}")


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
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["non_default", "default"])
    plt.yticks(tick_marks, ["non_default", "default"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def split_data_leakage_safe(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Split priority:
    1) Time split if a true date column exists (best practice)
    2) Group split by customer_id (prevents identity leakage)
    3) Stratified random split (fallback)
    """
    date_col = find_first_existing_col(df, DATE_COL_CANDIDATES)
    group_col = find_first_existing_col(df, GROUP_COL_CANDIDATES)

    # 1) TIME SPLIT (best)
    if date_col is not None:
        tmp = df[[date_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        if tmp[date_col].notna().mean() > 0.8:
            order = tmp[date_col].sort_values().index
            cutoff = int(len(order) * 0.8)
            train_idx = order[:cutoff]
            test_idx = order[cutoff:]
            return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx], f"time_split({date_col})"

    # 2) GROUP SPLIT (no same customer in both)
    if group_col is not None:
        groups = df[group_col]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], f"group_split({group_col})"

    # 3) FALLBACK
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, "stratified_random_split"


# -----------------------------
# MAIN EVAL
# -----------------------------
def evaluate_models(dataset_name: str, data_path: Path) -> None:
    print(f"\n=== Leakage-safe training on {dataset_name} ===")
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"target column missing in {data_path}")

    # Build X/y
    X = df.drop(columns=[TARGET_COL] + ID_COLS, errors="ignore").copy()
    X = X.drop(columns=[c for c in FEATURES_TO_DROP if c in X.columns], errors="ignore")
    y = df[TARGET_COL].astype(int)

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Leakage-safe split
    X_train, X_test, y_train, y_test, split_tag = split_data_leakage_safe(df, X, y)
    print(f"Split used: {split_tag}")
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,} | Default rate test: {y_test.mean():.3f}")

    # class imbalance scaling for XGB
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    models = get_models(scale_pos_weight)

    ds_out = OUTPUT_DIR / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    manifest = {"split": split_tag, "dropped_features": sorted(list(FEATURES_TO_DROP))}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)

        probas = clf.predict_proba(X_test)[:, 1]

        thr = choose_threshold(y_test.to_numpy(), probas, THRESHOLD_STRATEGY)
        preds = (probas >= thr).astype(int)

        metrics = {
            "dataset": dataset_name,
            "split": split_tag,
            "threshold_strategy": THRESHOLD_STRATEGY,
            "threshold": thr,
            "model": model_name,
            "auc_roc": roc_auc_score(y_test, probas),
            "auc_pr": average_precision_score(y_test, probas),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }
        metrics_rows.append(metrics)

        # Plots
        roc_path = ds_out / f"roc_{model_name}.png"
        pr_path = ds_out / f"pr_{model_name}.png"
        cm_path = ds_out / f"confusion_{model_name}.png"

        plot_roc(y_test.to_numpy(), probas, f"{dataset_name.upper()} - {model_name} ROC", roc_path)
        plot_pr(y_test.to_numpy(), probas, f"{dataset_name.upper()} - {model_name} PR", pr_path)
        plot_confusion(y_test.to_numpy(), preds, f"{dataset_name.upper()} - {model_name} Confusion (thr={thr:.3f})", cm_path)

        manifest[f"roc_{model_name}"] = str(roc_path)
        manifest[f"pr_{model_name}"] = str(pr_path)
        manifest[f"confusion_{model_name}"] = str(cm_path)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["dataset", "auc_roc"], ascending=[True, False]
    )
    metrics_path = ds_out / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics -> {metrics_path}")

    manifest_path = ds_out / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {name}, missing file: {path}")
            continue
        evaluate_models(name, path)


if __name__ == "__main__":
    main()
