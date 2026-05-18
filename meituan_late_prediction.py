# -*- coding: utf-8 -*-
"""
Meituan delivery lateness prediction and dispatch efficiency analysis.

Place the required CSV files under data/:
1. all_waybill_info_meituan_0322.csv
2. courier_wave_info_meituan.csv
3. dispatch_rider_meituan.csv
4. dispatch_waybill_meituan.csv

Run:
python meituan_late_prediction.py --data_dir data --output_dir outputs --fig_dir figures
"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def haversine_km(lng1, lat1, lng2, lat2):
    """Compute approximate distance in km from encoded longitude and latitude."""
    lng1 = lng1 / 1e6
    lat1 = lat1 / 1e6
    lng2 = lng2 / 1e6
    lat2 = lat2 / 1e6
    radius = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def count_list_like(value):
    """Count elements in strings such as '[1, 2, 3]'."""
    if pd.isna(value):
        return np.nan
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple, set)):
            return len(parsed)
        return np.nan
    except Exception:
        text = str(value).strip()
        if text in ("[]", ""):
            return 0
        return text.count(",") + 1


def load_and_build_features(data_dir: Path):
    waybill_path = data_dir / "all_waybill_info_meituan_0322.csv"
    rider_path = data_dir / "dispatch_rider_meituan.csv"

    print("Loading order/waybill data...")
    df = pd.read_csv(waybill_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()

    time_cols = [
        "estimate_arrived_time",
        "dispatch_time",
        "grab_time",
        "fetch_time",
        "arrive_time",
        "estimate_meal_prepare_time",
        "order_push_time",
        "platform_order_time",
    ]
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["is_courier_grabbed"] == 1) & (df["arrive_time"] > 0) & (df["dispatch_time"] > 0)].copy()

    df["late"] = (df["arrive_time"] > df["estimate_arrived_time"]).astype(int)

    df["dispatch_wait_min"] = (df["dispatch_time"] - df["order_push_time"]) / 60
    df["promised_remaining_min"] = (df["estimate_arrived_time"] - df["dispatch_time"]) / 60
    df["prep_remaining_min"] = (df["estimate_meal_prepare_time"] - df["dispatch_time"]) / 60
    df["estimated_prep_duration_min"] = (df["estimate_meal_prepare_time"] - df["platform_order_time"]) / 60

    df["grab_delay_min"] = (df["grab_time"] - df["dispatch_time"]) / 60
    df["fetch_after_grab_min"] = (df["fetch_time"] - df["grab_time"]) / 60
    df["delivery_after_fetch_min"] = (df["arrive_time"] - df["fetch_time"]) / 60
    df["total_delivery_min"] = (df["arrive_time"] - df["platform_order_time"]) / 60

    df["merchant_to_customer_km"] = haversine_km(
        df["sender_lng"], df["sender_lat"], df["recipient_lng"], df["recipient_lat"]
    )
    df["courier_to_merchant_km"] = haversine_km(
        df["grab_lng"], df["grab_lat"], df["sender_lng"], df["sender_lat"]
    )
    df["courier_to_customer_km"] = haversine_km(
        df["grab_lng"], df["grab_lat"], df["recipient_lng"], df["recipient_lat"]
    )

    dispatch_dt = pd.to_datetime(df["dispatch_time"], unit="s", errors="coerce")
    df["dispatch_hour"] = dispatch_dt.dt.hour
    df["dispatch_dayofweek"] = dispatch_dt.dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["dispatch_dayofweek"].isin([5, 6]).astype(int)

    if "is_prebook" not in df.columns:
        df["is_prebook"] = 0

    filter_cols = [
        "dispatch_wait_min",
        "promised_remaining_min",
        "prep_remaining_min",
        "estimated_prep_duration_min",
        "merchant_to_customer_km",
        "courier_to_merchant_km",
        "courier_to_customer_km",
    ]
    for col in filter_cols:
        low, high = df[col].quantile([0.005, 0.995])
        df = df[(df[col] >= low) & (df[col] <= high)]

    try:
        print("Loading rider state data...")
        riders = pd.read_csv(rider_path)
        riders = riders.loc[:, ~riders.columns.str.contains("^Unnamed")].copy()
        riders["onhand_order_count"] = riders["courier_waybills"].apply(count_list_like)
        riders = riders[["dt", "dispatch_time", "courier_id", "onhand_order_count"]]
        df = df.merge(riders, on=["dt", "dispatch_time", "courier_id"], how="left")
    except Exception as exc:
        print(f"Warning: rider state merge failed: {exc}")
        df["onhand_order_count"] = np.nan

    return df


def make_descriptive_outputs(df: pd.DataFrame, output_dir: Path, fig_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_model_records": int(len(df)),
        "late_rate": float(df["late"].mean()),
        "mean_total_delivery_min": float(df["total_delivery_min"].mean()),
        "mean_dispatch_wait_min": float(df["dispatch_wait_min"].mean()),
        "mean_merchant_to_customer_km": float(df["merchant_to_customer_km"].mean()),
        "rider_state_match_rate": float(df["onhand_order_count"].notna().mean()),
    }
    pd.Series(summary).to_csv(output_dir / "summary_metrics.csv", header=["value"])

    hourly = df.groupby("dispatch_hour")["late"].agg(["mean", "count"]).reset_index()
    hourly.to_csv(output_dir / "late_rate_by_hour.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(hourly["dispatch_hour"], hourly["mean"], marker="o")
    plt.xlabel("Dispatch hour")
    plt.ylabel("Late rate")
    plt.title("Late Rate by Dispatch Hour")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "01_late_rate_by_hour.png", dpi=200)
    plt.close()

    df["dispatch_wait_bin"] = pd.cut(df["dispatch_wait_min"], bins=[-1, 0, 1, 2, 5, 10, 20, 60], include_lowest=True)
    wait_bin = df.groupby("dispatch_wait_bin", observed=True)["late"].agg(["mean", "count"]).reset_index()
    wait_bin["dispatch_wait_bin"] = wait_bin["dispatch_wait_bin"].astype(str)
    wait_bin.to_csv(output_dir / "late_rate_by_dispatch_wait_bin.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.bar(wait_bin["dispatch_wait_bin"], wait_bin["mean"])
    plt.xlabel("Dispatch wait time bin, min")
    plt.ylabel("Late rate")
    plt.title("Late Rate by Dispatch Waiting Time")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "02_late_rate_by_dispatch_wait_bin.png", dpi=200)
    plt.close()

    quantiles = np.unique(df["merchant_to_customer_km"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values)
    df["distance_bin"] = pd.cut(df["merchant_to_customer_km"], bins=quantiles, include_lowest=True, duplicates="drop")
    dist_bin = df.groupby("distance_bin", observed=True)["late"].agg(["mean", "count"]).reset_index()
    dist_bin["distance_bin"] = dist_bin["distance_bin"].astype(str)
    dist_bin.to_csv(output_dir / "late_rate_by_distance_bin.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.bar(dist_bin["distance_bin"], dist_bin["mean"])
    plt.xlabel("Merchant-customer distance bin, km")
    plt.ylabel("Late rate")
    plt.title("Late Rate by Delivery Distance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "03_late_rate_by_distance_bin.png", dpi=200)
    plt.close()

    return summary


def train_models(df: pd.DataFrame, output_dir: Path, fig_dir: Path):
    feature_cols = [
        "dispatch_wait_min",
        "promised_remaining_min",
        "prep_remaining_min",
        "estimated_prep_duration_min",
        "merchant_to_customer_km",
        "courier_to_merchant_km",
        "courier_to_customer_km",
        "dispatch_hour",
        "dispatch_dayofweek",
        "is_weekend",
        "is_prebook",
        "da_id",
        "onhand_order_count",
    ]
    X = df[feature_cols].copy()
    y = df["late"].astype(int)

    if len(X) > 50000:
        sample_idx = X.sample(n=50000, random_state=42).index
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]

    categorical_features = ["dispatch_hour", "dispatch_dayofweek", "is_weekend", "is_prebook", "da_id"]
    numeric_features = [col for col in feature_cols if col not in categorical_features]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=10,
            min_samples_leaf=30,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    metrics = []
    best_name = None
    best_pipe = None
    best_auc = -1

    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
        }
        metrics.append(row)
        if row["auc"] > best_auc:
            best_name = name
            best_pipe = pipe
            best_auc = row["auc"]

        with open(output_dir / f"classification_report_{name}.txt", "w", encoding="utf-8") as file:
            file.write(classification_report(y_test, pred, digits=4))

    metrics_df = pd.DataFrame(metrics).sort_values("auc", ascending=False)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    best_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix: {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Not late", "Late"])
    plt.yticks([0, 1], ["Not late", "Late"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_confusion_matrix.png", dpi=200)
    plt.close()

    best_proba = best_pipe.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, best_proba)
    plt.title(f"ROC Curve: {best_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / "05_roc_curve.png", dpi=200)
    plt.close()

    rf_pipe = Pipeline([("preprocess", preprocessor), ("model", models["random_forest"])])
    rf_pipe.fit(X_train, y_train)
    model = rf_pipe.named_steps["model"]
    onehot = rf_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = onehot.get_feature_names_out(categorical_features)
    names = np.r_[numeric_features, cat_names]
    importance = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).head(15)
    importance.to_csv(output_dir / "feature_importance_top15.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importance")
    plt.tight_layout()
    plt.savefig(fig_dir / "06_feature_importance.png", dpi=200)
    plt.close()

    return metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--fig_dir", default="figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)

    df = load_and_build_features(data_dir)
    summary = make_descriptive_outputs(df, output_dir, fig_dir)
    metrics = train_models(df, output_dir, fig_dir)

    print("Done.")
    print("Summary metrics:")
    print(summary)
    print("Model metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
