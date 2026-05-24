# Task 4 - Pattern Identification (Clustering)
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we use UNSUPERVISED learning to find groups (clusters) of
# patients with similar profiles. We do not use the target variable here.
# The idea is to see if there are natural groups in the data and to
# describe them.
#
# We use the K-Means algorithm. To find a good number of clusters we
# use the "elbow method" (we run K-Means with different numbers of
# clusters and pick the value where the curve makes an "elbow").


import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClusteringConfig:
    input_path: Path = Path("../outputs/cleaned_dataset.csv")
    output_dir: Path = Path("../outputs")
    plot_dir: Path = Path("../outputs/plots")
    dpi: int = 130

    target_col: str = "weight_change_kg_6m"

    # Features used for clustering (no target, no IDs)
    clustering_features: List[str] = field(default_factory=lambda: [
        "age",
        "height_cm",
        "baseline_weight_kg",
        "baseline_bmi",
        "sleep_hours",
        "motivation_score",
        "mean_adherence_pct",
        "years_experience",
    ])

    k_range: List[int] = field(default_factory=lambda: list(range(2, 11)))
    chosen_k: int = 4
    random_state: int = 0


# ---------------------------------------------------------------------------
# FeatureScaler
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Wraps StandardScaler and keeps track of feature names."""

    def __init__(self):
        self._scaler = StandardScaler()
        self.feature_names: List[str] = []

    def fit_transform(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        self.feature_names = features
        X = df[features].copy()
        return self._scaler.fit_transform(X)


# ---------------------------------------------------------------------------
# KFinder – elbow + silhouette to pick k
# ---------------------------------------------------------------------------

class KFinder:
    """
    Runs K-Means for a range of k values and records inertia,
    silhouette score and Davies-Bouldin score to help choose k.
    """

    def __init__(self, config: ClusteringConfig):
        self.cfg = config
        self.results: List[dict] = []

    def run(self, X_scaled: np.ndarray) -> None:
        log.info("Scanning k values %s ...", self.cfg.k_range)
        for k in self.cfg.k_range:
            km = KMeans(n_clusters=k, random_state=self.cfg.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertia = km.inertia_
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            self.results.append({"k": k, "inertia": inertia, "silhouette": sil, "db": db})
            log.info(
                "  k=%-2d  inertia=%9.1f  silhouette=%+.3f  davies_bouldin=%.3f",
                k, inertia, sil, db,
            )

    def plot(self, out_path: Path) -> None:
        k_vals = [r["k"] for r in self.results]
        inertias = [r["inertia"] for r in self.results]
        sils = [r["silhouette"] for r in self.results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        ax1.plot(k_vals, inertias, marker="o", color="#3a7ebf")
        ax1.set_xlabel("Number of clusters (k)")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow method")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(k_vals, sils, marker="s", color="#e06c20")
        ax2.set_xlabel("Number of clusters (k)")
        ax2.set_ylabel("Silhouette score")
        ax2.set_title("Silhouette scores")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("Choosing the number of clusters", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# ClusterFitter
# ---------------------------------------------------------------------------

class ClusterFitter:
    """Fits K-Means with the chosen k and attaches labels to the DataFrame."""

    def __init__(self, config: ClusteringConfig):
        self.cfg = config
        self._model: Optional[KMeans] = None
        self.labels: Optional[np.ndarray] = None

    def fit(self, X_scaled: np.ndarray) -> np.ndarray:
        log.info("Fitting K-Means with k=%d", self.cfg.chosen_k)
        self._model = KMeans(
            n_clusters=self.cfg.chosen_k,
            random_state=self.cfg.random_state,
            n_init=10,
        )
        self.labels = self._model.fit_predict(X_scaled)
        inertia = self._model.inertia_
        sil = silhouette_score(X_scaled, self.labels)
        log.info("  Final inertia: %.1f  |  Silhouette: %.3f", inertia, sil)
        return self.labels

    def attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cluster"] = self.labels
        return df


# ---------------------------------------------------------------------------
# ClusterVisualiser
# ---------------------------------------------------------------------------

class ClusterVisualiser:
    """Creates PCA scatter, mean-profile bar charts and target breakdown."""

    def __init__(self, config: ClusteringConfig):
        self.cfg = config
        os.makedirs(config.plot_dir, exist_ok=True)

    def pca_scatter(self, X_scaled: np.ndarray, labels: np.ndarray) -> None:
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_
        log.info(
            "PCA: PC1=%.1f%%  PC2=%.1f%%  (total=%.1f%%)",
            var_explained[0] * 100, var_explained[1] * 100,
            sum(var_explained) * 100,
        )

        plt.figure(figsize=(8, 6))
        palette = plt.cm.tab10.colors
        for c in range(self.cfg.chosen_k):
            mask = labels == c
            plt.scatter(X2[mask, 0], X2[mask, 1],
                        label=f"Cluster {c}  (n={mask.sum()})",
                        alpha=0.55, s=25, color=palette[c])
        plt.xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)", fontsize=10)
        plt.ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)", fontsize=10)
        plt.title(f"K-Means clusters in 2D PCA space  (k={self.cfg.chosen_k})", fontsize=11)
        plt.legend(framealpha=0.7)
        plt.tight_layout()
        out = self.cfg.plot_dir / "clustering_pca.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)

    def profile_heatmap(self, profile: pd.DataFrame) -> None:
        """Normalised feature-mean heatmap so all features are comparable."""
        norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
        fig, ax = plt.subplots(figsize=(11, 3 + 0.4 * len(profile)))
        import seaborn as sns
        sns.heatmap(norm.T, annot=profile.T.round(1), fmt="g",
                    cmap="YlGnBu", linewidths=0.4, ax=ax,
                    annot_kws={"size": 9})
        ax.set_xlabel("Cluster")
        ax.set_title("Cluster profiles – normalised feature means\n(colour) with original values (text)")
        plt.tight_layout()
        out = self.cfg.plot_dir / "cluster_profiles_heatmap.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)

    def target_boxplot(self, df: pd.DataFrame) -> None:
        import seaborn as sns
        plt.figure(figsize=(6, 4))
        order = (
            df.groupby("cluster")[self.cfg.target_col]
            .mean()
            .sort_values()
            .index.tolist()
        )
        sns.boxplot(x="cluster", y=self.cfg.target_col, data=df,
                    order=order, palette="Blues")
        plt.title("Weight change per cluster", fontsize=11)
        plt.xlabel("Cluster")
        plt.ylabel("Weight change (kg)")
        plt.tight_layout()
        out = self.cfg.plot_dir / "cluster_weight_change.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# ClusterProfiler
# ---------------------------------------------------------------------------

class ClusterProfiler:
    """Prints and saves detailed statistics for each cluster."""

    def __init__(self, config: ClusteringConfig):
        self.cfg = config

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.cfg.clustering_features
        target = self.cfg.target_col

        log.info("=== Cluster sizes ===")
        sizes = df["cluster"].value_counts().sort_index()
        for c, n in sizes.items():
            log.info("  Cluster %d: %d rows (%.1f%%)", c, n, 100 * n / len(df))

        log.info("=== Feature means by cluster ===")
        profile = df.groupby("cluster")[features].mean().round(2)
        print(profile.to_string())
        print()

        log.info("=== Target '%s' by cluster ===", target)
        target_stats = (
            df.groupby("cluster")[target]
            .agg(["mean", "median", "std", "count"])
            .round(3)
        )
        print(target_stats.to_string())
        print()

        out = self.cfg.output_dir / "cluster_profiles.csv"
        profile.to_csv(out)
        log.info("Cluster profile saved to: %s", out)
        return profile


# ---------------------------------------------------------------------------
# Clustering orchestrator
# ---------------------------------------------------------------------------

class ClusteringRunner:
    def __init__(self, config: ClusteringConfig):
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
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        os.makedirs(self.cfg.plot_dir, exist_ok=True)

        # Scale features
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(self.df, self.cfg.clustering_features)
        log.info("Features scaled: %s", self.cfg.clustering_features)

        # Find k
        finder = KFinder(self.cfg)
        finder.run(X_scaled)
        finder.plot(self.cfg.plot_dir / "clustering_k_selection.png")

        # Fit chosen k
        fitter = ClusterFitter(self.cfg)
        labels = fitter.fit(X_scaled)
        self.df = fitter.attach_labels(self.df)

        # Profile
        profiler = ClusterProfiler(self.cfg)
        profile = profiler.run(self.df)

        # Visualise
        vis = ClusterVisualiser(self.cfg)
        vis.pca_scatter(X_scaled, labels)
        vis.profile_heatmap(profile)
        vis.target_boxplot(self.df)

        # Save labelled dataset
        out = self.cfg.output_dir / "clustered_dataset.csv"
        self.df.to_csv(out, index=False)
        log.info("Clustered dataset saved to: %s", out)

        log.info("Task 4 complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = ClusteringConfig()
    ClusteringRunner(cfg).run()


if __name__ == "__main__":
    main()
