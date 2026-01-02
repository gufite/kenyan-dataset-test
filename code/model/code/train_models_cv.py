from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_FOLDS = 3  # keep runtime reasonable while still checking stability
# File is repo/code/model/code/train_models_cv.py -> repo root is three levels up
ROOT = Path(__file__).resolve().parents[3]
DATASETS: Dict[str, Path] = {
    "full": ROOT / "data/feature-generated/kenya_engineered_features.csv",
    "borrower": ROOT / "data/feature-generated/kenya_engineered_features_borrower_side.csv",
}
OUTPUT_DIR = ROOT / "code" / "model" / "outputs_cv"
TARGET_COL = "target"
ID_COLS = ["customer_id", "tbl_loan_id", "pseudo_disb_date"]
GROUP_COL_CANDIDATES = ["customer_id", "customerId", "client_id"]
DATE_COL_CANDIDATES = ["pseudo_disb_date", "disb_date", "disbursement_date", "application_date", "loan_date"]
FEATURES_TO_DROP = {
    "interest_rate",
    "repayment_intensity",
    "lender_risk_profile",
    "pseudo_disb_date",  # used only for time split, not for modeling
}


@dataclass
class SplitPlan:
    splitter: object
    split_tag: str
    order_index: Optional[pd.Index] = None  # used for time splits to maintain order


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


def find_first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_split_plan(df: pd.DataFrame) -> SplitPlan:
    """
    Priority:
    1) TimeSeriesSplit if a usable date column exists.
    2) GroupKFold if a group column exists.
    3) StratifiedKFold fallback.
    """
    date_col = find_first_existing_col(df, DATE_COL_CANDIDATES)
    group_col = find_first_existing_col(df, GROUP_COL_CANDIDATES)

    if date_col is not None:
        tmp = df[[date_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        if tmp[date_col].notna().mean() > 0.8:
            order = tmp[date_col].sort_values().index
            tss = TimeSeriesSplit(n_splits=N_FOLDS)
            return SplitPlan(splitter=tss, split_tag=f"time_cv({date_col})", order_index=order)

    if group_col is not None:
        gkf = GroupKFold(n_splits=N_FOLDS)
        return SplitPlan(splitter=gkf, split_tag=f"group_cv({group_col})", order_index=None)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return SplitPlan(splitter=skf, split_tag="stratified_cv", order_index=None)


def get_models(scale_pos_weight: float) -> Dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
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
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "catboost": CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            random_seed=RANDOM_STATE,
        ),
    }


def evaluate_cv(dataset_name: str, data_path: Path) -> None:
    print(f"\n=== CV eval on {dataset_name} ===")
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"target column missing in {data_path}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL] + ID_COLS, errors="ignore").copy()
    X = X.drop(columns=[c for c in FEATURES_TO_DROP if c in X.columns], errors="ignore")

    split_plan = build_split_plan(df)
    print(f"Split plan: {split_plan.split_tag}")

    # Apply ordering for time splits
    if split_plan.order_index is not None:
        X = X.loc[split_plan.order_index].reset_index(drop=True)
        y = y.loc[split_plan.order_index].reset_index(drop=True)
        groups = None
    else:
        groups = df[find_first_existing_col(df, GROUP_COL_CANDIDATES)] if find_first_existing_col(df, GROUP_COL_CANDIDATES) else None

    preprocessor, _, _ = build_preprocessor(X)
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
    models = get_models(scale_pos_weight)

    ds_out = OUTPUT_DIR / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)

    records = []
    for model_name, model in models.items():
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            split_plan.splitter.split(X, y, groups=groups)  # type: ignore[arg-type]
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_test)[:, 1]
            preds = (probas >= 0.5).astype(int)

            fold_metrics.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "split": split_plan.split_tag,
                    "fold": fold_idx,
                    "auc_roc": roc_auc_score(y_test, probas),
                    "auc_pr": average_precision_score(y_test, probas),
                    "accuracy": accuracy_score(y_test, preds),
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall": recall_score(y_test, preds, zero_division=0),
                    "f1": f1_score(y_test, preds, zero_division=0),
                }
            )

        # Aggregate per model
        agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        agg["dataset"] = dataset_name
        agg["model"] = model_name
        agg["split"] = split_plan.split_tag
        agg["fold"] = "mean"
        fold_metrics.append(agg)
        records.extend(fold_metrics)

    metrics_df = pd.DataFrame(records)
    metrics_path = ds_out / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    manifest = {
        "split": split_plan.split_tag,
        "n_folds": N_FOLDS,
        "dropped_features": sorted(list(FEATURES_TO_DROP)),
    }
    (ds_out / "artifacts.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved CV metrics -> {metrics_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {name}, missing file: {path}")
            continue
        evaluate_cv(name, path)


if __name__ == "__main__":
    main()
