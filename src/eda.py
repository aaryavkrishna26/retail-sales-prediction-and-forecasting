"""
Exploratory Data Analysis (EDA) plotting and analytical diagnostic utilities.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import FIGURES_DIR

# Set clean aesthetic styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
sns.set_theme(style="whitegrid", palette="muted")


def plot_overall_sales_trend(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Plot aggregate daily total sales over the entire historical observation period."""
    daily_sales = df.groupby("date")["sales"].sum().reset_index()

    plt.figure(figsize=(14, 5))
    plt.plot(daily_sales["date"], daily_sales["sales"], color="#2563EB", linewidth=1.1, label="Daily Total Sales")
    
    # Highlight major macro event: 2016-04-16 Ecuador Earthquake
    quake_date = pd.to_datetime("2016-04-16")
    plt.axvline(quake_date, color="#EF4444", linestyle="--", alpha=0.8, label="Ecuador Earthquake (Apr 2016)")

    plt.title("Total Retail Sales Over Time (Corporación Favorita 2013-2017)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Total Units Sold", fontsize=11)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "daily_sales_trend.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_top_families(df: pd.DataFrame, top_n: int = 10, save_path: Optional[Path] = None) -> None:
    """Plot top-selling product families ranked by total volume."""
    family_sales = (
        df.groupby("family", observed=False)["sales"]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=family_sales,
        x="sales",
        y="family",
        hue="family",
        palette="Blues_r",
        legend=False,
    )
    plt.title(f"Top {top_n} Product Families by Total Sales Volume", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Total Sales Volume (Units)", fontsize=11)
    plt.ylabel("Product Family", fontsize=11)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "top_families.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_payday_impact(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Analyze sales distribution difference between regular days and paydays (15th and month-end)."""
    data = df.copy()
    if "is_payday" not in data.columns:
        day = data["date"].dt.day
        is_end = data["date"].dt.is_month_end
        data["is_payday"] = ((day == 15) | is_end).astype(int)

    # Calculate average daily sales per store on payday vs non-payday
    store_daily = data.groupby(["store_nbr", "date", "is_payday"])["sales"].sum().reset_index()
    store_daily["payday_label"] = store_daily["is_payday"].map({0: "Regular Days", 1: "Paydays (15th & Month-End)"})

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        x="payday_label",
        y="sales",
        hue="payday_label",
        data=store_daily,
        showfliers=False,
        palette=["#94A3B8", "#10B981"],
        legend=False,
    )
    plt.title("Store-Level Daily Sales: Regular Days vs. Paydays (15th & Month-End)", fontsize=12, weight="bold", pad=12)
    plt.xlabel("")
    plt.ylabel("Total Daily Store Sales (Units)", fontsize=11)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "payday_impact.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_day_of_week_seasonality(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Analyze intra-week seasonality across days of the week."""
    data = df.copy()
    data["dayofweek"] = data["date"].dt.day_name()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    daily_totals = data.groupby(["date", "dayofweek"])["sales"].sum().reset_index()

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x="dayofweek",
        y="sales",
        hue="dayofweek",
        data=daily_totals,
        order=days_order,
        palette="crest",
        showfliers=False,
        legend=False,
    )
    plt.title("Weekly Seasonality: Sales Distribution by Day of Week", fontsize=12, weight="bold", pad=12)
    plt.xlabel("Day of Week", fontsize=11)
    plt.ylabel("Total Aggregate Units Sold", fontsize=11)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "day_of_week_seasonality.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_store_types_and_clusters(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Analyze sales distribution across store types (A through E) and clusters."""
    merged = sales_df.groupby("store_nbr")["sales"].sum().reset_index()
    merged = merged.merge(stores_df, on="store_nbr", how="left")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Store Type Sales
    type_data = merged.groupby("type", observed=False)["sales"].mean().reset_index()
    type_order = sorted(merged["type"].dropna().unique())
    sns.barplot(
        data=type_data,
        x="type",
        y="sales",
        hue="type",
        palette="Blues_r",
        order=type_order,
        legend=False,
        ax=ax1,
    )
    ax1.set_title("Average Total Sales per Store by Store Type", fontsize=11, weight="bold")
    ax1.set_xlabel("Store Type", fontsize=10)
    ax1.set_ylabel("Mean Sales per Store (Units)", fontsize=10)

    # Store Cluster Sales
    cluster_data = merged.groupby("cluster", observed=False)["sales"].mean().reset_index().sort_values("cluster")
    sns.barplot(
        data=cluster_data,
        x="cluster",
        y="sales",
        hue="cluster",
        palette="viridis",
        legend=False,
        ax=ax2,
    )
    ax2.set_title("Average Total Sales per Store by Cluster", fontsize=11, weight="bold")
    ax2.set_xlabel("Store Cluster (1-17)", fontsize=10)
    ax2.set_ylabel("Mean Sales per Store (Units)", fontsize=10)

    plt.tight_layout()
    out_file = save_path or (FIGURES_DIR / "store_types_and_clusters.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_oil_price_correlation(
    sales_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Plot weekly aggregate sales against Brent crude oil price fluctuations."""
    cleaned_oil = oil_df.copy().set_index("date").asfreq("D")
    cleaned_oil["dcoilwtico"] = cleaned_oil["dcoilwtico"].interpolate(method="linear").bfill().ffill()

    weekly_sales = (
        sales_df.groupby(pd.Grouper(key="date", freq="W-MON"))["sales"]
        .sum()
        .reset_index()
    )
    weekly_oil = (
        cleaned_oil.reset_index().groupby(pd.Grouper(key="date", freq="W-MON"))["dcoilwtico"]
        .mean()
        .reset_index()
    )
    weekly = weekly_sales.merge(weekly_oil, on="date", how="inner")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    color_sales = "#1E40AF"
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Weekly Total Sales", color=color_sales, fontsize=11)
    ax1.plot(weekly["date"], weekly["sales"], color=color_sales, linewidth=1.3, label="Weekly Sales")
    ax1.tick_params(axis="y", labelcolor=color_sales)

    ax2 = ax1.twinx()
    color_oil = "#D97706"
    ax2.set_ylabel("Crude Oil Price ($/barrel)", color=color_oil, fontsize=11)
    ax2.plot(weekly["date"], weekly["dcoilwtico"], color=color_oil, linestyle="--", linewidth=1.2, label="Oil Price")
    ax2.tick_params(axis="y", labelcolor=color_oil)

    plt.title("Weekly Retail Sales vs. Macroeconomic Oil Price Trend (Ecuador)", fontsize=13, weight="bold", pad=12)
    fig.tight_layout()

    out_file = save_path or (FIGURES_DIR / "oil_sales_relationship.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_promotion_impact(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Analyze correlation between promotional items count and sales volume."""
    daily_promo = (
        df.groupby("date")[["sales", "onpromotion"]]
        .sum()
        .reset_index()
    )
    
    # Group days into Zero, Moderate, and High promotional activity
    daily_promo["onpromotion_tier"] = "Zero Promotions"
    promo_days = daily_promo["onpromotion"] > 0
    if promo_days.any():
        median_promo = daily_promo.loc[promo_days, "onpromotion"].median()
        daily_promo.loc[promo_days & (daily_promo["onpromotion"] <= median_promo), "onpromotion_tier"] = "Moderate Promotion"
        daily_promo.loc[daily_promo["onpromotion"] > median_promo, "onpromotion_tier"] = "High Promotion"

    order_tiers = ["Zero Promotions", "Moderate Promotion", "High Promotion"]

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="onpromotion_tier",
        y="sales",
        hue="onpromotion_tier",
        data=daily_promo,
        order=order_tiers,
        palette="Purples",
        showfliers=False,
        legend=False,
    )
    plt.title("Daily Network-Wide Sales Volume by Promotion Intensity Tier", fontsize=12, weight="bold", pad=12)
    plt.xlabel("Promotional Activity Tier", fontsize=11)
    plt.ylabel("Total Daily Units Sold", fontsize=11)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "promotion_sales_impact.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
