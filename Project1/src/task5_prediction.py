# Task 5 - Prediction Models
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we use SUPERVISED learning to predict the variable
# weight_change_kg_6m (how many kilos the patient lost or gained
# after 6 months on the diet program).
#
# We try several models, compare them, and pick the best one. We
# also look at which features the best model thinks are important.


import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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
class PredictionConfig:
    input_path: Path = Path("../outputs/cleaned_dataset.csv")
    output_dir: Path = Path("../outputs")
    plot_dir: Path = Path("../outputs/plots")
    dpi: int = 130

    target_col: str = "weight_change_kg_6m"

    id_columns: List[str] = field(default_factory=lambda: [
        "program_id",
        "patient_id",
        "nutritionist_id",
        "diet_id",
        "record_created_at",
    ])

    # Split proportions: 60% train, 20% validation, 20% test
    test_size: float = 0.20
    val_fraction_of_remaining: float = 0.25   # 0.25 * 0.80 = 0.20 total
    random_state: int = 42

    cv_folds: int = 5


# ---------------------------------------------------------------------------
# DataPreparer
# ---------------------------------------------------------------------------

class DataPreparer:
    """
    Loads the cleaned dataset, drops ID columns, one-hot encodes categoricals,
    and splits into train / validation / test.
    """

    def __init__(self, config: PredictionConfig):
        self.cfg = config

    def prepare(self) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.Series,   pd.Series,   pd.Series,
    ]:
        if not self.cfg.input_path.exists():
            log.error("File not found: %s", self.cfg.input_path)
            sys.exit(1)

        df = pd.read_csv(self.cfg.input_path)
        log.info("Loaded dataset  shape=%s", df.shape)

        # Drop IDs and any leftover cluster column
        drop_cols = [c for c in self.cfg.id_columns + ["cluster"]
                     if c in df.columns]
        df = df.drop(columns=drop_cols)

        X = df.drop(columns=[self.cfg.target_col])
        y = df[self.cfg.target_col]

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)
        log.info("Features after one-hot encoding: %d", X.shape[1])

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )

        # Second split: train vs validation from the remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.cfg.val_fraction_of_remaining,
            random_state=self.cfg.random_state,
        )

        total = len(X)
        log.info(
            "Split -> train=%d (%.0f%%)  val=%d (%.0f%%)  test=%d (%.0f%%)",
            len(X_train), 100 * len(X_train) / total,
            len(X_val),   100 * len(X_val)   / total,
            len(X_test),  100 * len(X_test)  / total,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Returns a dictionary of name -> sklearn pipeline for each model."""

    @staticmethod
    def build() -> Dict[str, Any]:
        models: Dict[str, Any] = {}

        # Ridge regression (regularised linear model – good baseline)
        models["Ridge Regression"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

        # Support Vector Regression with RBF kernel
        models["SVR (RBF)"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")),
        ])

        # K-Nearest Neighbours
        models["KNN (k=7)"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ])

        # Decision Tree  (moderately constrained to avoid total overfitting)
        models["Decision Tree"] = Pipeline([
            ("model", DecisionTreeRegressor(
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            )),
        ])

        # Random Forest
        models["Random Forest"] = Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        # Gradient Boosting
        models["Gradient Boosting"] = Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )),
        ])

        return models


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Fits each model, evaluates on validation set and records metrics."""

    def __init__(self, config: PredictionConfig):
        self.cfg = config
        self.records: List[dict] = []
        self.fitted_models: Dict[str, Any] = {}

    @staticmethod
    def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    def run(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> pd.DataFrame:

        log.info("=== Training and validation ===")
        for name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            self.fitted_models[name] = pipeline

            val_pred = pipeline.predict(X_val)
            m = self._metrics(y_val, val_pred)

            # Cross-validation R² on training data (extra robustness check)
            cv_r2 = cross_val_score(
                pipeline, X_train, y_train,
                cv=self.cfg.cv_folds, scoring="r2", n_jobs=-1,
            )
            m["CV_R2_mean"] = round(cv_r2.mean(), 4)
            m["CV_R2_std"] = round(cv_r2.std(), 4)
            m["model"] = name
            self.records.append(m)
            log.info(
                "  %-22s  val MAE=%6.3f  RMSE=%6.3f  R2=%+.3f  CV_R2=%+.3f±%.3f",
                name, m["MAE"], m["RMSE"], m["R2"],
                m["CV_R2_mean"], m["CV_R2_std"],
            )

        results = pd.DataFrame(self.records).set_index("model")
        return results

    def best_model_name(self, results: pd.DataFrame) -> str:
        # Pick the model with the highest validation R²
        best = results["R2"].idxmax()
        log.info("Best model on validation set: '%s'  R2=%.4f", best, results.loc[best, "R2"])
        return best


# ---------------------------------------------------------------------------
# TestEvaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    """Re-evaluates the best model on the held-out test set."""

    def __init__(self, config: PredictionConfig):
        self.cfg = config

    def run(
        self,
        model: Any,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        log.info(
            "=== Test set results for '%s' ===\n"
            "  MAE  = %.4f\n  RMSE = %.4f\n  R2   = %.4f",
            model_name, mae, rmse, r2,
        )
        return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2,
                "y_pred": y_pred}


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyser
# ---------------------------------------------------------------------------

class FeatureImportanceAnalyser:
    """
    Extracts feature importances for tree-based models.
    Falls back to a note if the model type does not expose importances.
    """

    def __init__(self, config: PredictionConfig):
        self.cfg = config
        os.makedirs(config.plot_dir, exist_ok=True)

    def run(self, pipeline: Any, feature_names: List[str]) -> None:
        # Navigate into the pipeline to find the underlying estimator
        if hasattr(pipeline, "named_steps"):
            estimator = pipeline.named_steps.get("model", pipeline)
        else:
            estimator = pipeline

        if not hasattr(estimator, "feature_importances_"):
            log.info("Model does not expose feature_importances_; skipping plot.")
            return

        importances = pd.Series(estimator.feature_importances_, index=feature_names)
        top = importances.sort_values(ascending=False).head(20)

        log.info("=== Top 10 most important features ===")
        for feat, imp in top.head(10).items():
            bar = "█" * int(imp * 200)
            log.info("  %-35s  %.4f  %s", feat, imp, bar)

        plt.figure(figsize=(9, 6))
        top[::-1].plot.barh(color="#3a7ebf", edgecolor="white")
        plt.title("Top 20 most important features", fontsize=12)
        plt.xlabel("Importance (mean decrease in impurity)")
        plt.tight_layout()
        out = self.cfg.plot_dir / "feature_importance.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# PredictionPlotter
# ---------------------------------------------------------------------------

class PredictionPlotter:
    """Predicted vs actual and residuals plots."""

    def __init__(self, config: PredictionConfig):
        self.cfg = config
        os.makedirs(config.plot_dir, exist_ok=True)

    def pred_vs_actual(self, y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # --- Predicted vs actual ---
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax1.scatter(y_test, y_pred, alpha=0.35, s=20, color="#3a7ebf")
        ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Ideal")
        ax1.set_xlabel("Actual weight change (kg)")
        ax1.set_ylabel("Predicted weight change (kg)")
        ax1.set_title(f"Predicted vs actual\n({model_name})")
        ax1.legend()

        # --- Residuals ---
        residuals = y_pred - y_test.values
        ax2.scatter(y_pred, residuals, alpha=0.35, s=20, color="#e06c20")
        ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax2.set_xlabel("Predicted value")
        ax2.set_ylabel("Residual  (predicted − actual)")
        ax2.set_title("Residual plot")

        plt.suptitle(model_name, fontsize=12, y=1.01)
        plt.tight_layout()
        out = self.cfg.plot_dir / "predicted_vs_actual.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)

    def model_comparison_bar(self, results: pd.DataFrame) -> None:
        metrics = ["MAE", "RMSE", "R2"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = plt.cm.Set2.colors
        for ax, metric in zip(axes, metrics):
            vals = results[metric].sort_values(ascending=(metric != "R2"))
            bars = ax.barh(vals.index, vals.values,
                           color=colors[:len(vals)], edgecolor="white")
            ax.set_title(metric)
            ax.set_xlabel(metric)
            for bar, val in zip(bars, vals.values):
                ax.text(bar.get_width() + 0.001 * abs(vals.values).max(),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)
        plt.suptitle("Model comparison on validation set", fontsize=12)
        plt.tight_layout()
        out = self.cfg.plot_dir / "model_comparison.png"
        plt.savefig(out, dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Prediction orchestrator
# ---------------------------------------------------------------------------

class PredictionRunner:
    def __init__(self, config: PredictionConfig):
        self.cfg = config

    def run(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        os.makedirs(self.cfg.plot_dir, exist_ok=True)

        # Data
        preparer = DataPreparer(self.cfg)
        X_train, X_val, X_test, y_train, y_val, y_test = preparer.prepare()

        feature_names: List[str] = list(X_train.columns)

        # Models
        models = ModelFactory.build()

        # Train + validate
        evaluator = ModelEvaluator(self.cfg)
        val_results = evaluator.run(models, X_train, y_train, X_val, y_val)

        # Save validation comparison
        out_csv = self.cfg.output_dir / "model_comparison.csv"
        val_results.to_csv(out_csv)
        log.info("Validation comparison saved to: %s", out_csv)

        # Pick best model
        best_name = evaluator.best_model_name(val_results)
        best_pipeline = evaluator.fitted_models[best_name]

        # Test evaluation
        tester = TestEvaluator(self.cfg)
        test_result = tester.run(best_pipeline, best_name, X_test, y_test)

        # Plots
        plotter = PredictionPlotter(self.cfg)
        plotter.model_comparison_bar(val_results)
        plotter.pred_vs_actual(y_test, test_result["y_pred"], best_name)

        # Feature importance
        FeatureImportanceAnalyser(self.cfg).run(best_pipeline, feature_names)

        log.info("Task 5 complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = PredictionConfig()
    PredictionRunner(cfg).run()


if __name__ == "__main__":
    main()
