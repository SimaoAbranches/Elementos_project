# Task 2 - Data Cleaning and Processing
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we clean the merged dataset from Task 1. The main steps are:
#   1. Remove columns that are duplicated (the same information).
#   2. Find and fix outliers (values that are impossible in real life).
#   3. Fill in missing values.
#   4. Save the cleaned dataset for the next tasks.


import pandas as pd
import numpy as np


# ---- Step 1: load the merged dataset from Task 1 ---------------------------

data = pd.read_csv("../outputs/merged_dataset.csv")
print("Initial shape:", data.shape)
print("Initial missing values:", data.isna().sum().sum())
print()


# ---- Step 2: remove redundant (duplicated) columns -------------------------

# I checked the data and found 4 columns that contain the same information
# as another column. Keeping both is not useful and can confuse the models.
#
# - "bmi_redundant" is the same as "baseline_bmi" (and it also has some
#   wrong values like 9410, so it is better to drop it).
# - "experience_years" is the same as "years_experience".
# - "adherence_ratio" is the same as "mean_adherence_pct" / 100.
# - "total_macros" is the sum of carb_pct + protein_pct + fat_pct, so it
#   does not bring new information.

columns_to_drop = [
    "bmi_redundant",
    "experience_years",
    "adherence_ratio",
    "total_macros",
]
data = data.drop(columns=columns_to_drop)
print("After dropping redundant columns:", data.shape)
print("Dropped columns:", columns_to_drop)
print()


# ---- Step 3: standardize text in categorical columns -----------------------

# Some categorical columns have the same value written in different ways
# (for example "F", " F", "f", "female" all mean female). I make them
# lowercase, remove extra spaces and map synonyms to one common form.

print("Standardizing text columns:")

# Sex: keep only "f" or "m"
data["sex"] = data["sex"].astype(str).str.strip().str.lower()
data["sex"] = data["sex"].replace({"female": "f", "male": "m"})
print("  sex unique values after cleaning:", sorted(data["sex"].dropna().unique()))

# Approach (nutritionist): lowercase + strip
data["approach"] = data["approach"].astype(str).str.strip().str.lower()
data["approach"] = data["approach"].replace({"nan": pd.NA})
print("  approach unique values after cleaning:", sorted([x for x in data["approach"].dropna().unique()]))

# Diet type, diet name, specialty: just strip and lowercase to be safe
for col in ["diet_type", "diet_name", "specialty"]:
    data[col] = data[col].astype(str).str.strip()

print()


# ---- Step 4: fix impossible values (outliers) ------------------------------

# Some columns have values that cannot exist in real life. For example,
# a person cannot be 10 cm tall or sleep -1 hours. I replace these wrong
# values with NaN so they are treated as missing values and filled later.
#
# The realistic ranges below come from common sense and from the context
# of the problem (adults that follow a diet program).

print("Fixing impossible values:")

# age: between 5 and 100 years
mask = (data["age"] < 5) | (data["age"] > 100)
print("  age outside [5, 100]:", mask.sum())
data.loc[mask, "age"] = np.nan

# height: between 100 cm and 220 cm
mask = (data["height_cm"] < 100) | (data["height_cm"] > 220)
print("  height_cm outside [100, 220]:", mask.sum())
data.loc[mask, "height_cm"] = np.nan

# weight: between 30 kg and 250 kg
mask = (data["baseline_weight_kg"] < 30) | (data["baseline_weight_kg"] > 250)
print("  baseline_weight_kg outside [30, 250]:", mask.sum())
data.loc[mask, "baseline_weight_kg"] = np.nan

# sleep hours: between 0 and 24
mask = (data["sleep_hours"] < 0) | (data["sleep_hours"] > 24)
print("  sleep_hours outside [0, 24]:", mask.sum())
data.loc[mask, "sleep_hours"] = np.nan

# motivation score: between 0 and 1
mask = (data["motivation_score"] < 0) | (data["motivation_score"] > 1)
print("  motivation_score outside [0, 1]:", mask.sum())
data.loc[mask, "motivation_score"] = np.nan

# years of experience: between 0 and 50
mask = (data["years_experience"] < 0) | (data["years_experience"] > 50)
print("  years_experience outside [0, 50]:", mask.sum())
data.loc[mask, "years_experience"] = np.nan

# adherence percentage: between 0 and 100
mask = (data["mean_adherence_pct"] < 0) | (data["mean_adherence_pct"] > 100)
print("  mean_adherence_pct outside [0, 100]:", mask.sum())
data.loc[mask, "mean_adherence_pct"] = np.nan

# weight change after 6 months: between -50 and +50 kg is realistic
mask = (data["weight_change_kg_6m"] < -50) | (data["weight_change_kg_6m"] > 50)
print("  weight_change_kg_6m outside [-50, 50]:", mask.sum())
data.loc[mask, "weight_change_kg_6m"] = np.nan

print()


# ---- Step 5: drop rows where the target value is missing -------------------

# The variable we want to predict is "weight_change_kg_6m" (how many kilos
# the patient lost or gained after 6 months). If this value is missing we
# cannot use the row for training a model, so we drop those rows.

before = data.shape[0]
data = data.dropna(subset=["weight_change_kg_6m"])
print("Rows dropped because target was missing:", before - data.shape[0])
print()


# ---- Step 6: fill remaining missing values ---------------------------------

# Numeric columns: fill with the median (median is more robust to outliers
# than the mean).
# Categorical columns: fill with the mode (the most frequent value).

print("Filling missing values:")
for col in data.columns:
    if data[col].isna().sum() == 0:
        continue
    if data[col].dtype == "object":
        # Categorical column - fill with the most frequent value
        most_frequent = data[col].mode()[0]
        data[col] = data[col].fillna(most_frequent)
        print(f"  {col} (categorical) -> filled with '{most_frequent}'")
    else:
        # Numeric column - fill with the median
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value)
        print(f"  {col} (numeric) -> filled with median {median_value:.2f}")

print()


# ---- Step 7: final check ---------------------------------------------------

print("Final shape:", data.shape)
print("Remaining missing values:", data.isna().sum().sum())
print()


# ---- Step 8: save the cleaned dataset --------------------------------------

output_path = "../outputs/cleaned_dataset.csv"
data.to_csv(output_path, index=False)
print("Cleaned dataset saved to:", output_path)
