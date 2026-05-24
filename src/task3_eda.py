# Task 3 - Exploratory Data Analysis (EDA)
# In this task we explore the cleaned dataset to understand it. We look at:
#   - the basic statistics of each column
#   - the distribution of each variable (histograms, boxplots)
#   - how the variables relate to each other (correlation)
#   - how the target "weight_change_kg_6m" is influenced by sex, diet,
#     nutritionist approach and other features
#
# The plots are saved in outputs/plots/ so they can be added to the report.


import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# Configuration
@dataclass
class EDAConfig:
    input_path: Path = Path("../outputs/cleaned_dataset.csv")
    plot_dir: Path = Path("../outputs/plots")
    dpi: int = 130

    target_col: str = "weight_change_kg_6m"

    numeric_features: List[str] = field(default_factory=lambda: [
        "age",
        "height_cm",
        "baseline_weight_kg",
        "baseline_bmi",
        "sleep_hours",
        "motivation_score",
        "mean_adherence_pct",
        "weight_change_kg_6m",
        "years_experience",
    ])

    categorical_features: List[str] = field(default_factory=lambda: [
        "sex",
        "diet_type",
        "diet_name",
        "approach",
        "specialty",
        "smoker",
    ])


# Descriptive statistics
class DescriptiveAnalyser:
    """Prints summary statistics for numeric and categorical columns."""

    def __init__(self, df: pd.DataFrame, cfg: EDAConfig):
        self.df = df
        self.cfg = cfg

    def run(self) -> None:
        log.info("=== Descriptive statistics (numeric) ===")
        stats = self.df[self.cfg.numeric_features].describe().round(2)
        print(stats.to_string())
        print()

        log.info("=== Value counts (categorical) ===")
        for col in self.cfg.categorical_features:
            if col not in self.df.columns:
                log.warning("Column '%s' not found.", col)
                continue
            counts = self.df[col].value_counts()
            pcts = (counts / len(self.df) * 100).round(1)
            summary = pd.DataFrame({"count": counts, "pct": pcts})
            log.info("  %s:\n%s\n", col, summary.to_string())


# Distribution plots
class DistributionPlotter:
    """Histograms and boxplots for numeric features."""

    def __init__(self, df: pd.DataFrame, cfg: EDAConfig):
        self.df = df
        self.cfg = cfg
        os.makedirs(cfg.plot_dir, exist_ok=True)

    def plot_histograms(self) -> None:
        cols = self.cfg.numeric_features
        n_cols = 3
        n_rows = -(-len(cols) // n_cols)  # ceiling division
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes_flat = axes.flat
        for ax, col in zip(axes_flat, cols):
            ax.hist(self.df[col].dropna(), bins=25, color="#3a7ebf", edgecolor="white", linewidth=0.5)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in list(axes_flat)[len(cols):]:
            ax.set_visible(False)
        plt.suptitle("Distribution of numeric variables", fontsize=13, y=1.01)
        plt.tight_layout()
        out = self.cfg.plot_dir / "histograms.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)

    def plot_boxplots(self) -> None:
        cols = self.cfg.numeric_features
        n_cols = 3
        n_rows = -(-len(cols) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        for ax, col in zip(axes.flat, cols):
            ax.boxplot(self.df[col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#cce0f5"))
            ax.set_title(col, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in list(axes.flat)[len(cols):]:
            ax.set_visible(False)
        plt.suptitle("Boxplots of numeric variables", fontsize=13, y=1.01)
        plt.tight_layout()
        out = self.cfg.plot_dir / "boxplots.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)


# Correlation analysis
class CorrelationAnalyser:
    """Pearson correlation matrix and ranking vs target."""

    def __init__(self, df: pd.DataFrame, cfg: EDAConfig):
        self.df = df
        self.cfg = cfg
        os.makedirs(cfg.plot_dir, exist_ok=True)

    def run(self) -> None:
        corr = self.df[self.cfg.numeric_features].corr()

        # Heatmap
        plt.figure(figsize=(10, 8))
        mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, mask=mask, linewidths=0.4,
            annot_kws={"size": 8},
        )
        plt.title("Correlation matrix – numeric variables", fontsize=12)
        plt.tight_layout()
        out = self.cfg.plot_dir / "correlation_heatmap.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)

        # Print correlations sorted by target
        target = self.cfg.target_col
        ranked = corr[target].drop(labels=[target]).sort_values()
        log.info("=== Correlation with target '%s' ===", target)
        for var, val in ranked.items():
            bar = "+" * int(abs(val) * 20) if val >= 0 else "-" * int(abs(val) * 20)
            log.info("  %-30s  %+.3f  %s", var, val, bar)


# Categorical breakdown
class CategoricalBreakdown:
    """
    For each categorical feature, plots a boxplot of the target
    and prints group means.
    """

    def __init__(self, df: pd.DataFrame, cfg: EDAConfig):
        self.df = df
        self.cfg = cfg
        os.makedirs(cfg.plot_dir, exist_ok=True)

    def run(self) -> None:
        target = self.cfg.target_col
        for col in self.cfg.categorical_features:
            if col not in self.df.columns:
                log.warning("Column '%s' not found; skipping.", col)
                continue

            # Group means
            means = (
                self.df.groupby(col)[target]
                .agg(["mean", "median", "count"])
                .round(3)
                .sort_values("mean")
            )
            log.info("=== Target breakdown by '%s' ===\n%s\n", col, means.to_string())

            # Boxplot
            n_cats = self.df[col].nunique()
            fig_w = max(6, n_cats * 1.4)
            order = (
                self.df.groupby(col)[target]
                .mean()
                .sort_values()
                .index.tolist()
            )
            plt.figure(figsize=(fig_w, 4))
            sns.boxplot(
                x=col, y=target, hue=col, data=self.df, order=order,
                palette="Blues_d", legend=False,
            )
            plt.title(f"Weight change by {col}", fontsize=11)
            plt.xticks(rotation=30, ha="right")
            plt.xlabel("")
            plt.tight_layout()
            fname = f"weight_change_by_{col}.png"
            out = self.cfg.plot_dir / fname
            plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
            plt.close()
            log.info("Saved: %s", out)


# Scatter: adherence vs target
class AdherenceScatter:
    """Scatterplot of adherence vs weight change with regression line."""

    def __init__(self, df: pd.DataFrame, cfg: EDAConfig):
        self.df = df
        self.cfg = cfg
        os.makedirs(cfg.plot_dir, exist_ok=True)

    def run(self) -> None:
        x_col = "mean_adherence_pct"
        y_col = self.cfg.target_col
        if x_col not in self.df.columns:
            log.warning("Column '%s' not found; skipping scatter.", x_col)
            return

        plt.figure(figsize=(7, 5))
        sns.regplot(
            x=self.df[x_col], y=self.df[y_col],
            scatter_kws={"alpha": 0.3, "s": 20},
            line_kws={"color": "crimson"},
        )
        plt.xlabel("Adherence (%)")
        plt.ylabel("Weight change (kg) after 6 months")
        plt.title("Relationship between adherence and weight change")
        plt.tight_layout()
        out = self.cfg.plot_dir / "adherence_vs_weight_change.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)


# EDA orchestrator
class EDARunner:
    def __init__(self, config: EDAConfig):
        self.cfg = config
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        if not self.cfg.input_path.exists():
            log.error("File not found: %s", self.cfg.input_path)
            sys.exit(1)
        self.df = pd.read_csv(self.cfg.input_path)
        log.info("Loaded dataset  shape=%s", self.df.shape)

    def run(self) -> None:
        self.load()

        DescriptiveAnalyser(self.df, self.cfg).run()

        dp = DistributionPlotter(self.df, self.cfg)
        dp.plot_histograms()
        dp.plot_boxplots()

        CorrelationAnalyser(self.df, self.cfg).run()
        CategoricalBreakdown(self.df, self.cfg).run()
        AdherenceScatter(self.df, self.cfg).run()

        log.info("Task 3 complete. All plots saved to: %s", self.cfg.plot_dir)


# Main
def main() -> None:
    cfg = EDAConfig()
    EDARunner(cfg).run()


if __name__ == "__main__":
    main()


