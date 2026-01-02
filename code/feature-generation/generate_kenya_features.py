from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw-data/KQ_Transformed_Final12.csv")
OUT_PATH = Path("data/feature-generated/kenya_engineered_features.csv")
BORROWER_OUT_PATH = Path("data/feature-generated/kenya_engineered_features_borrower_side.csv")

# Borrower-side features as defined in the Kenya sheet of the feature dictionary.
BORROWER_FEATURES = [
    "num_previous_loans",
    "avg_time_bw_loans",
    "avg_past_amount",
    "avg_past_daily_burden",
    "std_past_amount",
    "std_past_daily_burden",
    "trend_in_amount",
    "trend_in_burden",
    "Total_Amount",
    "daily_burden",
    "amount_ratio",
    "burden_ratio",
    "amount_bucket",
    "burden_percentile",
    "borrower_history_strength",
    "month",
    "quarter",
    "week_of_year",
    "days_to_salary_day",
    "days_to_local_festival",
    "account_age_days",
    "loan_frequency_per_year",
    "latest_amount_ma3",
]


def build_pseudo_disb_date(disb_month: int, disb_dow: int) -> pd.Timestamp:
    """
    Construct a deterministic proxy disbursement date using the provided month and day-of-week.
    We anchor all loans in 2023 and choose the first occurrence of the requested weekday.
    """
    base = dt.date(2023, int(disb_month), 1)
    delta_days = (int(disb_dow) - base.weekday()) % 7
    return pd.Timestamp(base + dt.timedelta(days=delta_days))


def slope_from_history(values: Iterable[float]) -> pd.Series:
    """
    Compute the slope of a simple linear regression over past observations only.
    Returns NaN for the first two loans (insufficient history).
    """
    slopes: list[float] = []
    history: list[float] = []
    for value in values:
        if len(history) < 2:
            slopes.append(np.nan)
        else:
            x = np.arange(len(history))
            y = np.asarray(history, dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
        history.append(value)
    return pd.Series(slopes, index=getattr(values, "index", None))


def quantile_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    """Bucket a numeric series using quantiles with a safe fallback to equal-width bins."""
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        return pd.cut(series, bins=len(labels), labels=labels)


def days_until_salary(date: pd.Timestamp, salary_day: int = 25) -> int:
    """
    Approximate days until the next salary date (defaults to 25th of each month).
    If the salary day for the current month has passed, roll forward to the next month.
    """
    next_salary = pd.Timestamp(date.year, date.month, salary_day)
    if date > next_salary:
        year = date.year + 1 if date.month == 12 else date.year
        month = 1 if date.month == 12 else date.month + 1
        next_salary = pd.Timestamp(year, month, salary_day)
    return int((next_salary - date).days)


def days_until_local_festival(date: pd.Timestamp, month: int = 12, day: int = 12) -> int:
    """
    Proxy for distance to a local festival (defaults to Jamhuri Day on Dec 12).
    Rolls to the following year if the date has already passed.
    """
    festival = pd.Timestamp(date.year, month, day)
    if date > festival:
        festival = pd.Timestamp(date.year + 1, month, day)
    return int((festival - date).days)


def main() -> None:
    raw_df = pd.read_csv(RAW_PATH)
    kenya = raw_df[raw_df["country_id"] == "Kenya"].copy()
    if kenya.empty:
        raise SystemExit("No Kenya rows found in the raw dataset.")

    # Derive borrower id and an ordering proxy for time.
    kenya["customer_id"] = kenya["ID"].str.extract(r"ID_(\d{6})")[0]
    kenya["pseudo_disb_date"] = kenya.apply(
        lambda row: build_pseudo_disb_date(row["disb_month"], row["disb_dow"]),
        axis=1,
    )

    # Deduplicate by exact loan id and near-identical borrower/amount/duration combos.
    before = len(kenya)
    kenya = kenya.drop_duplicates(subset=["tbl_loan_id"])
    after_id_drop = len(kenya)
    kenya = kenya.drop_duplicates(
        subset=["customer_id", "Total_Amount", "duration", "target", "disb_month", "disb_dow"]
    )
    after_near_drop = len(kenya)
    print(
        f"Deduplicated Kenya rows: {before:,} -> {after_id_drop:,} after tbl_loan_id drop,"
        f" -> {after_near_drop:,} after near-duplicate drop."
    )

    kenya = kenya.sort_values(
        ["customer_id", "pseudo_disb_date", "tbl_loan_id", "lender_id"]
    ).reset_index(drop=True)

    # Core pricing and burden metrics.
    kenya["daily_burden"] = kenya["Total_Amount_to_Repay"] / kenya["duration"]

    cust_group = kenya.groupby("customer_id", group_keys=False)

    kenya["num_previous_loans"] = cust_group.cumcount()
    kenya["num_previous_defaults"] = cust_group["target"].apply(
        lambda s: s.shift().expanding().sum()
    )
    kenya["past_default_rate"] = kenya["num_previous_defaults"] / kenya[
        "num_previous_loans"
    ].replace(0, np.nan)
    kenya["past_default_rate"] = kenya["past_default_rate"].fillna(0)

    kenya["avg_past_amount"] = cust_group["Total_Amount"].apply(
        lambda s: s.shift().expanding().mean()
    )
    kenya["avg_past_daily_burden"] = cust_group["daily_burden"].apply(
        lambda s: s.shift().expanding().mean()
    )
    kenya["std_past_amount"] = cust_group["Total_Amount"].apply(
        lambda s: s.shift().expanding().std(ddof=0)
    )
    kenya["std_past_daily_burden"] = cust_group["daily_burden"].apply(
        lambda s: s.shift().expanding().std(ddof=0)
    )

    kenya["amount_ratio"] = kenya["Total_Amount"] / kenya["avg_past_amount"].replace(
        0, np.nan
    )
    kenya.loc[kenya["num_previous_loans"] == 0, "amount_ratio"] = 1.0
    kenya["burden_ratio"] = kenya["daily_burden"] / kenya[
        "avg_past_daily_burden"
    ].replace(0, np.nan)
    kenya.loc[kenya["num_previous_loans"] == 0, "burden_ratio"] = 1.0

    # Time since prior loans (relative to the proxy disbursement date).
    gaps = cust_group["pseudo_disb_date"].diff().dt.days
    kenya["days_since_last_loan"] = gaps
    kenya["avg_time_bw_loans"] = gaps.groupby(
        kenya["customer_id"], group_keys=False
    ).apply(
        lambda s: s.shift().expanding().mean()
    )

    kenya["trend_in_amount"] = cust_group["Total_Amount"].apply(slope_from_history)
    kenya["trend_in_burden"] = cust_group["daily_burden"].apply(slope_from_history)

    kenya["duration_bucket"] = pd.cut(
        kenya["duration"],
        bins=[-np.inf, 7, 14, 30, 60, np.inf],
        labels=["<=1w", "<=2w", "<=1m", "<=2m", ">2m"],
    )
    kenya["amount_bucket"] = quantile_bucket(
        kenya["Total_Amount"], labels=["q1", "q2", "q3", "q4"]
    )
    kenya["burden_percentile"] = kenya["daily_burden"].rank(pct=True)

    kenya["month"] = kenya["disb_month"]
    kenya["quarter"] = ((kenya["disb_month"] - 1) // 3) + 1
    kenya["week_of_year"] = kenya["pseudo_disb_date"].dt.isocalendar().week.astype(int)

    kenya["days_to_salary_day"] = kenya["pseudo_disb_date"].apply(days_until_salary)
    kenya["days_to_local_festival"] = kenya["pseudo_disb_date"].apply(
        days_until_local_festival
    )

    kenya["lender_exposure_ratio"] = (
        kenya["Amount_Funded_By_Lender"] / kenya["Total_Amount"]
    )

    first_disb = cust_group["pseudo_disb_date"].transform("min")
    kenya["account_age_days"] = (kenya["pseudo_disb_date"] - first_disb).dt.days

    total_loans_to_date = kenya["num_previous_loans"] + 1
    kenya["loan_frequency_per_year"] = total_loans_to_date / (
        kenya["account_age_days"].clip(lower=1) / 365.25
    )
    kenya.loc[kenya["account_age_days"] == 0, "loan_frequency_per_year"] = 0

    kenya["repayment_consistency"] = 1 - cust_group["target"].apply(
        lambda s: s.shift().expanding().mean()
    )
    kenya["repayment_consistency"] = kenya["repayment_consistency"].fillna(1)

    kenya["latest_amount_ma3"] = cust_group["Total_Amount"].apply(
        lambda s: s.shift().rolling(window=3, min_periods=1).mean()
    )

    max_loans = kenya["num_previous_loans"].max()
    loan_factor = kenya["num_previous_loans"] / max_loans if max_loans else 0
    trend_scale_amount = kenya["Total_Amount"].median() or 1
    trend_scale_burden = kenya["daily_burden"].median() or 1
    trend_factor = 0.5 + 0.5 * np.tanh(
        (kenya["trend_in_amount"].fillna(0) / trend_scale_amount)
        + (kenya["trend_in_burden"].fillna(0) / trend_scale_burden)
    )
    kenya["borrower_history_strength"] = (
        0.4 * loan_factor.fillna(0)
        + 0.3 * (1 - kenya["past_default_rate"])
        + 0.3 * kenya["repayment_consistency"]
    ) * trend_factor
    kenya["borrower_history_strength"] = kenya["borrower_history_strength"].clip(
        lower=0
    )

    feature_columns = [
        "num_previous_loans",
        "num_previous_defaults",
        "past_default_rate",
        "days_since_last_loan",
        "avg_time_bw_loans",
        "avg_past_amount",
        "avg_past_daily_burden",
        "std_past_amount",
        "std_past_daily_burden",
        "trend_in_amount",
        "trend_in_burden",
        "Total_Amount",
        "Total_Amount_to_Repay",
        "duration",
        "daily_burden",
        "amount_ratio",
        "burden_ratio",
        "duration_bucket",
        "amount_bucket",
        "burden_percentile",
        "borrower_history_strength",
        "month",
        "quarter",
        "week_of_year",
        "days_to_salary_day",
        "days_to_local_festival",
        "lender_id",
        "lender_exposure_ratio",
        "account_age_days",
        "loan_frequency_per_year",
        "repayment_consistency",
        "latest_amount_ma3",
    ]

    output_columns = [
        "customer_id",
        "tbl_loan_id",
        "pseudo_disb_date",
    ] + feature_columns + ["target"]
    engineered = kenya[output_columns]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(OUT_PATH, index=False)
    print(f"Built {len(engineered):,} Kenya feature rows -> {OUT_PATH}")

    borrower_columns = [
        "customer_id",
        "tbl_loan_id",
        "pseudo_disb_date",
    ] + BORROWER_FEATURES + ["target"]
    missing = set(borrower_columns) - set(engineered.columns)
    if missing:
        raise SystemExit(f"Missing expected borrower columns: {sorted(missing)}")
    engineered[borrower_columns].to_csv(BORROWER_OUT_PATH, index=False)
    print(
        f"Built {len(engineered):,} borrower-side Kenya feature rows -> {BORROWER_OUT_PATH}"
    )


if __name__ == "__main__":
    main()
