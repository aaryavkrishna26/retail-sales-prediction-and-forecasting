"""
Exploratory Data Analysis (EDA) plotting and analytical diagnostic utilities.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import FIGURES_DIR

# Set clean aesthetic style
sns.set_theme(style="whitegrid", palette="muted")


def plot_overall_sales_trend(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Plot aggregate daily total sales over the entire historical observation period."""
    daily_sales = df.groupby("date")["sales"].sum().reset_index()

    plt.figure(figsize=(14, 5))
    plt.plot(daily_sales["date"], daily_sales["sales"], color="#2563EB", linewidth=1.2, label="Daily Total Sales")
    plt.title("Total Retail Sales Over Time (Corporación Favorita)", fontsize=14, weight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Total Units Sold", fontsize=11)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "daily_sales_trend.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_top_families(df: pd.DataFrame, top_n: int = 10, save_path: Optional[Path] = None) -> None:
    """Plot top-selling product families ranked by total volume."""
    family_sales = (
        df.groupby("family")["sales"]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=family_sales, x="sales", y="family", palette="Blues_r")
    plt.title(f"Top {top_n} Product Families by Total Sales Volume", fontsize=13, weight="bold")
    plt.xlabel("Total Sales Volume", fontsize=11)
    plt.ylabel("Product Family", fontsize=11)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "top_families.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_payday_impact(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Analyze sales distribution difference between paydays and non-paydays."""
    if "is_payday" not in df.columns:
        return

    plt.figure(figsize=(7, 5))
    sns.boxplot(x="is_payday", y="sales", data=df, showfliers=False, palette=["#94A3B8", "#10B981"])
    plt.title("Sales Distribution: Regular Days vs. Paydays (15th & Month-End)", fontsize=12, weight="bold")
    plt.xticks([0, 1], ["Regular Days", "Paydays"])
    plt.xlabel("")
    plt.ylabel("Unit Sales (Excl. Outliers)", fontsize=11)
    plt.tight_layout()

    out_file = save_path or (FIGURES_DIR / "payday_impact.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_oil_price_correlation(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Plot weekly aggregate sales against Brent crude oil price fluctuations."""
    if "dcoilwtico" not in df.columns:
        return

    weekly = (
        df.groupby(pd.Grouper(key="date", freq="W-MON"))
        .agg({"sales": "sum", "dcoilwtico": "mean"})
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(14, 5))
    color_sales = "#1E40AF"
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Weekly Sales Volume", color=color_sales, fontsize=11)
    ax1.plot(weekly["date"], weekly["sales"], color=color_sales, label="Weekly Sales")
    ax1.tick_params(axis="y", labelcolor=color_sales)

    ax2 = ax1.twinx()
    color_oil = "#D97706"
    ax2.set_ylabel("Crude Oil Price ($/barrel)", color=color_oil, fontsize=11)
    ax2.plot(weekly["date"], weekly["dcoilwtico"], color=color_oil, linestyle="--", label="Oil Price")
    ax2.tick_params(axis="y", labelcolor=color_oil)

    plt.title("Weekly Store Sales vs. Macroeconomic Oil Price Trend", fontsize=13, weight="bold")
    fig.tight_layout()

    out_file = save_path or (FIGURES_DIR / "oil_sales_relationship.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
