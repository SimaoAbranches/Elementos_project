# Task 1 - Data Integration
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we read the four CSV files and join them into one big table.
# The four files have information about:
#   - patients
#   - diets
#   - nutritionists
#   - outcomes (the result of each diet program)


# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Configuration

@dataclass
class IntegrationConfig:
    data_dir: Path = Path("../data")
    output_dir: Path = Path("../outputs")
    output_file: str = "merged_dataset.csv"

    # Maps each logical table to a filename
    csv_files: Dict[str, str] = field(default_factory=lambda: {
        "patients": "patients.csv",
        "diets": "diets.csv",
        "nutritionists": "nutritionists.csv",
        "outcomes": "outcomes.csv",
    })

    # Join keys for each table against the central outcomes table
    join_keys: Dict[str, str] = field(default_factory=lambda: {
        "patients": "patient_id",
        "diets": "diet_id",
        "nutritionists": "nutritionist_id",
    })


# DataLoader
class DataLoader:
    """Reads all CSV files and stores them as DataFrames."""

    def __init__(self, config: IntegrationConfig):
        self.cfg = config
        self.tables: Dict[str, pd.DataFrame] = {}

    def load_all(self) -> None:
        for name, filename in self.cfg.csv_files.items():
            path = self.cfg.data_dir / filename
            if not path.exists():
                log.error("File not found: %s", path)
                sys.exit(1)
            df = pd.read_csv(path)
            self.tables[name] = df
            log.info("Loaded %-15s  shape=%s", name, df.shape)

    def summary(self) -> None:
        log.info("--- Table summary ---")
        for name, df in self.tables.items():
            missing = df.isna().sum().sum()
            log.info("  %-15s  rows=%-6d  cols=%-3d  missing=%d",
                     name, len(df), len(df.columns), missing)


# DataIntegrator
class DataIntegrator:
    """Merges multiple tables into a single flat DataFrame."""

    def __init__(self, loader: DataLoader, config: IntegrationConfig):
        self.loader = loader
        self.cfg = config
        self.merged: Optional[pd.DataFrame] = None

    def merge(self) -> pd.DataFrame:
        tables = self.loader.tables
        outcomes = tables["outcomes"]
        log.info("Starting merge from outcomes table (%d rows)", len(outcomes))

        result = outcomes.copy()
        for table_name, key in self.cfg.join_keys.items():
            before = len(result)
            result = pd.merge(result, tables[table_name], on=key, how="left")
            after = len(result)
            if before != after:
                log.warning(
                    "Row count changed after merging '%s': %d -> %d",
                    table_name, before, after,
                )
            else:
                log.info("Merged '%-15s' on '%-20s'  rows still %d", table_name, key, after)

        self._integrity_check(outcomes, result)
        self.merged = result
        return result

    def _integrity_check(self, outcomes: pd.DataFrame, merged: pd.DataFrame) -> None:
        log.info("--- Integrity checks ---")
        if len(merged) == len(outcomes):
            log.info("Row count preserved: %d rows (OK)", len(merged))
        else:
            log.warning("Row count mismatch! outcomes=%d  merged=%d", len(outcomes), len(merged))

        total_missing = merged.isna().sum().sum()
        log.info("Total missing values in merged table: %d", total_missing)

        missing_per_col = merged.isna().sum()
        cols_with_missing = missing_per_col[missing_per_col > 0]
        if not cols_with_missing.empty:
            log.info("Columns with missing values:")
            for col, n in cols_with_missing.items():
                pct = 100.0 * n / len(merged)
                log.info("  %-30s  %d  (%.1f%%)", col, n, pct)

    def save(self) -> None:
        if self.merged is None:
            raise RuntimeError("Call merge() before save().")
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        out = self.cfg.output_dir / self.cfg.output_file
        self.merged.to_csv(out, index=False)
        log.info("Merged dataset saved to: %s  shape=%s", out, self.merged.shape)

    def print_columns(self) -> None:
        if self.merged is None:
            return
        log.info("Columns in merged dataset (%d total):", len(self.merged.columns))
        for c in self.merged.columns:
            dtype = str(self.merged[c].dtype)
            log.info("  %-35s  dtype=%s", c, dtype)


# Main
def main() -> None:
    cfg = IntegrationConfig()

    loader = DataLoader(cfg)
    loader.load_all()
    loader.summary()

    integrator = DataIntegrator(loader, cfg)
    integrator.merge()
    integrator.print_columns()
    integrator.save()

    log.info("Task 1 complete.")


if __name__ == "__main__":
    main()

