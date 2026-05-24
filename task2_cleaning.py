# Task 2 - Data Cleaning and Processing
# In this task we clean the merged dataset from Task 1. The main steps are:
#   1. Remove columns that are duplicated (the same information).
#   2. Find and fix outliers (values that are impossible in real life).
#   3. Fill in missing values.
#   4. Save the cleaned dataset for the next tasks.


import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
class CleaningConfig:
    input_path: Path = Path("../outputs/merged_dataset.csv")
    output_path: Path = Path("../outputs/cleaned_dataset.csv")
    target_col: str = "weight_change_kg_6m"

    redundant_columns: List[str] = field(default_factory=lambda: [
        "bmi_redundant",
        "experience_years",
        "adherence_ratio",
        "total_macros",
    ])

    # (column, min_value, max_value) – values outside range become NaN
    domain_bounds: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("age",                   5,    100),
        ("height_cm",           100,    220),
        ("baseline_weight_kg",   30,    250),
        ("sleep_hours",           0,     24),
        ("motivation_score",      0,      1),
        ("years_experience",      0,     50),
        ("mean_adherence_pct",    0,    100),
        ("weight_change_kg_6m", -50,     50),
    ])

    # Categorical columns and the standardisation to apply
    text_columns: Dict[str, Optional[Dict[str, str]]] = field(default_factory=lambda: {
        "sex": {"female": "f", "male": "m", "0": "f", "1": "m"},
        "approach": {},
        "diet_type": {},
        "diet_name": {},
        "specialty": {},
    })


# ColumnDropper
class ColumnDropper:
    """Removes redundant or uninformative columns."""

    def __init__(self, columns: List[str]):
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in self.columns if c in df.columns]
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            log.warning("Columns to drop not found (already absent): %s", missing)
        df = df.drop(columns=existing)
        log.info("Dropped %d redundant columns: %s", len(existing), existing)
        return df


# TextStandardiser
class TextStandardiser:
    """Normalises categorical columns: lowercase, strip whitespace, map synonyms."""

    def __init__(self, column_map: Dict[str, Optional[Dict[str, str]]]):
        self.column_map = column_map

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, synonyms in self.column_map.items():
            if col not in df.columns:
                log.warning("Column '%s' not found; skipping standardisation.", col)
                continue
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace("nan", pd.NA)
            if synonyms:
                df[col] = df[col].replace(synonyms)
            unique_vals = sorted(df[col].dropna().unique().tolist())
            log.info("  %-20s  unique values after cleaning: %s", col, unique_vals)
        return df


# OutlierHandler
class OutlierHandler:
    """Replaces values outside realistic domain bounds with NaN."""

    def __init__(self, bounds: List[Tuple[str, float, float]]):
        self.bounds = bounds

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Checking domain bounds:")
        total_replaced = 0
        for col, lo, hi in self.bounds:
            if col not in df.columns:
                log.warning("  Column '%s' not found; skipping.", col)
                continue
            mask = (df[col] < lo) | (df[col] > hi)
            count = mask.sum()
            total_replaced += count
            if count > 0:
                df.loc[mask, col] = np.nan
                log.info("  %-30s  [%g, %g]  replaced=%d", col, lo, hi, count)
            else:
                log.info("  %-30s  [%g, %g]  all values OK", col, lo, hi)
        log.info("Total values replaced with NaN: %d", total_replaced)
        return df


# MissingValueFiller
class MissingValueFiller:
    """
    Fills missing values.
    Numeric columns  -> median
    Categorical cols -> mode
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_summary = df.isna().sum()
        cols_to_fill = missing_summary[missing_summary > 0].index.tolist()

        if not cols_to_fill:
            log.info("No missing values to fill.")
            return df

        log.info("Filling missing values (%d columns):", len(cols_to_fill))
        for col in cols_to_fill:
            n_missing = df[col].isna().sum()
            if df[col].dtype == "object" or str(df[col].dtype) == "category":
                fill_val = df[col].mode(dropna=True).iloc[0]
                strategy = "mode"
            else:
                fill_val = df[col].median()
                strategy = "median"
            df[col] = df[col].fillna(fill_val)
            log.info("  %-30s  strategy=%-6s  fill_value=%-10s  n_filled=%d",
                     col, strategy, str(fill_val)[:10], n_missing)
        return df


# DataCleaner
class DataCleaner:
    """Runs the full cleaning pipeline on the merged dataset."""

    def __init__(self, config: CleaningConfig):
        self.cfg = config
        self.df: Optional[pd.DataFrame] = None

        self._dropper = ColumnDropper(config.redundant_columns)
        self._standardiser = TextStandardiser(config.text_columns)
        self._outlier_handler = OutlierHandler(config.domain_bounds)
        self._filler = MissingValueFiller()

    # helpers
    def _load(self) -> None:
        if not self.cfg.input_path.exists():
            log.error("Input file not found: %s", self.cfg.input_path)
            sys.exit(1)
        self.df = pd.read_csv(self.cfg.input_path)
        log.info("Loaded dataset  shape=%s  missing=%d",
                 self.df.shape, self.df.isna().sum().sum())

    def _drop_missing_target(self) -> None:
        before = len(self.df)
        self.df = self.df.dropna(subset=[self.cfg.target_col])
        dropped = before - len(self.df)
        log.info("Dropped %d rows with missing target '%s'",
                 dropped, self.cfg.target_col)

    def _save(self) -> None:
        os.makedirs(self.cfg.output_path.parent, exist_ok=True)
        self.df.to_csv(self.cfg.output_path, index=False)
        log.info("Cleaned dataset saved to: %s  shape=%s",
                 self.cfg.output_path, self.df.shape)

    # pipeline
    def run(self) -> pd.DataFrame:
        self._load()

        log.info("Step 1: Drop redundant columns")
        self.df = self._dropper.apply(self.df)

        log.info("Step 2: Standardise text columns")
        self.df = self._standardiser.apply(self.df)

        log.info("Step 3: Fix impossible values")
        self.df = self._outlier_handler.apply(self.df)

        log.info("Step 4: Drop rows with missing target")
        self._drop_missing_target()

        log.info("Step 5: Fill remaining missing values")
        self.df = self._filler.apply(self.df)

        log.info("Final checks")
        log.info("Final shape: %s", self.df.shape)
        remaining = self.df.isna().sum().sum()
        if remaining == 0:
            log.info("No missing values remaining (OK)")
        else:
            log.warning("%d missing values still present!", remaining)

        self._save()
        return self.df


# Main
def main() -> None:
    cfg = CleaningConfig()
    cleaner = DataCleaner(cfg)
    cleaner.run()
    log.info("Task 2 complete.")


if __name__ == "__main__":
    main()
