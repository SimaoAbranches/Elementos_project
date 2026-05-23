# Task 3 - Exploratory Data Analysis (EDA)
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we explore the cleaned dataset to understand it. We look at:
#   - the basic statistics of each column
#   - the distribution of each variable (histograms, boxplots)
#   - how the variables relate to each other (correlation)
#   - how the target "weight_change_kg_6m" is influenced by sex, diet,
#     nutritionist approach and other features
#
# The plots are saved in outputs/plots/ so they can be added to the report.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---- Step 1: load the cleaned dataset --------------------------------------

data = pd.read_csv("../outputs/cleaned_dataset.csv")
print("Dataset shape:", data.shape)
print()


# ---- Step 2: descriptive statistics ----------------------------------------

# describe() gives count, mean, std, min, max and quartiles for each
# numeric column.
print("--- Descriptive statistics (numeric columns) ---")
print(data.describe().round(2))
print()

# For categorical columns we just count the unique values.
print("--- Categorical columns ---")
for col in data.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(data[col].value_counts())


# ---- Step 3: plot distributions of the main numeric variables --------------

# Histograms show how the values are spread.
numeric_cols = [
    "age",
    "height_cm",
    "baseline_weight_kg",
    "baseline_bmi",
    "sleep_hours",
    "motivation_score",
    "mean_adherence_pct",
    "weight_change_kg_6m",
    "years_experience",
]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, col in zip(axes.flat, numeric_cols):
    ax.hist(data[col], bins=20, color="steelblue", edgecolor="black")
    ax.set_title(col)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig("../outputs/plots/histograms.png", dpi=120)
plt.close()
print("Saved: histograms.png")


# ---- Step 4: correlation matrix --------------------------------------------

# The correlation tells us if two numeric variables move together.
# Values close to 1 or -1 mean a strong relation. Values close to 0
# mean no linear relation.

corr = data[numeric_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix of numeric variables")
plt.tight_layout()
plt.savefig("../outputs/plots/correlation_heatmap.png", dpi=120)
plt.close()
print("Saved: correlation_heatmap.png")

# Print the top correlations with the target.
target = "weight_change_kg_6m"
print(f"\n--- Correlation with target '{target}' ---")
print(corr[target].sort_values())


# ---- Step 5: target by categorical variables -------------------------------

# Question: does the SEX of the patient influence the weight change?
plt.figure(figsize=(6, 4))
sns.boxplot(x="sex", y="weight_change_kg_6m", data=data)
plt.title("Weight change by sex")
plt.tight_layout()
plt.savefig("../outputs/plots/weight_change_by_sex.png", dpi=120)
plt.close()
print("Saved: weight_change_by_sex.png")
print("\nMean weight change by sex:")
print(data.groupby("sex")["weight_change_kg_6m"].mean().round(2))

# Question: which DIET TYPE has better results?
plt.figure(figsize=(8, 4))
sns.boxplot(x="diet_type", y="weight_change_kg_6m", data=data)
plt.title("Weight change by diet type")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../outputs/plots/weight_change_by_diet_type.png", dpi=120)
plt.close()
print("Saved: weight_change_by_diet_type.png")
print("\nMean weight change by diet_type:")
print(data.groupby("diet_type")["weight_change_kg_6m"].mean().round(2))

# Question: which DIET NAME has better results?
plt.figure(figsize=(10, 4))
order = data.groupby("diet_name")["weight_change_kg_6m"].mean().sort_values().index
sns.boxplot(x="diet_name", y="weight_change_kg_6m", data=data, order=order)
plt.title("Weight change by diet name")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../outputs/plots/weight_change_by_diet_name.png", dpi=120)
plt.close()
print("Saved: weight_change_by_diet_name.png")
print("\nMean weight change by diet_name:")
print(data.groupby("diet_name")["weight_change_kg_6m"].mean().sort_values().round(2))

# Question: does the NUTRITIONIST APPROACH matter?
plt.figure(figsize=(7, 4))
sns.boxplot(x="approach", y="weight_change_kg_6m", data=data)
plt.title("Weight change by nutritionist approach")
plt.tight_layout()
plt.savefig("../outputs/plots/weight_change_by_approach.png", dpi=120)
plt.close()
print("Saved: weight_change_by_approach.png")
print("\nMean weight change by approach:")
print(data.groupby("approach")["weight_change_kg_6m"].mean().round(2))

# Question: does the SPECIALTY of the nutritionist matter?
print("\nMean weight change by specialty:")
print(data.groupby("specialty")["weight_change_kg_6m"].mean().round(2))

# Question: does being a SMOKER influence the result?
print("\nMean weight change by smoker:")
print(data.groupby("smoker")["weight_change_kg_6m"].mean().round(2))


# ---- Step 6: relation between adherence and result -------------------------

# A natural hypothesis: if the patient follows the diet better
# (high adherence), the weight change is bigger.
plt.figure(figsize=(6, 4))
plt.scatter(data["mean_adherence_pct"], data["weight_change_kg_6m"], alpha=0.3)
plt.xlabel("mean_adherence_pct")
plt.ylabel("weight_change_kg_6m")
plt.title("Adherence vs weight change")
plt.tight_layout()
plt.savefig("../outputs/plots/adherence_vs_weight.png", dpi=120)
plt.close()
print("\nSaved: adherence_vs_weight.png")

print("\nDone. All plots are in outputs/plots/")
