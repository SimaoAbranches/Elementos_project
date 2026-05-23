# Task 1 - Data Integration
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we read the four CSV files and join them into one big table.
# The four files have information about:
#   - patients
#   - diets
#   - nutritionists
#   - outcomes (the result of each diet program)


import pandas as pd


# ---- Step 1: read the four CSV files ---------------------------------------

# Each CSV is loaded into a DataFrame (a table in pandas).
patients = pd.read_csv("../data/patients.csv")
diets = pd.read_csv("../data/diets.csv")
nutritionists = pd.read_csv("../data/nutritionists.csv")
outcomes = pd.read_csv("../data/outcomes.csv")

# Print the size of each table to make sure the files were read correctly.
print("Patients table:", patients.shape)
print("Diets table:", diets.shape)
print("Nutritionists table:", nutritionists.shape)
print("Outcomes table:", outcomes.shape)
print()


# ---- Step 2: join the tables -----------------------------------------------

# The "outcomes" table is the central one. Each row is a diet program and
# it has the IDs that link to the other three tables:
#   - patient_id        -> patients table
#   - diet_id           -> diets table
#   - nutritionist_id   -> nutritionists table
#
# So we start from "outcomes" and add the other tables one by one.
# We use how="left" to keep every row of outcomes even if some ID has no
# match. This way we can see later if there are missing links.

data = pd.merge(outcomes, patients, on="patient_id", how="left")
data = pd.merge(data, diets, on="diet_id", how="left")
data = pd.merge(data, nutritionists, on="nutritionist_id", how="left")


# ---- Step 3: check the result ----------------------------------------------

print("Final merged table:", data.shape)
print("Number of columns:", len(data.columns))
print()
print("Column names:")
for col in data.columns:
    print(" -", col)
print()

# Check that no rows were lost during the merge.
# The number of rows must be the same as the outcomes table.
if data.shape[0] == outcomes.shape[0]:
    print("OK - no rows were lost during the merge.")
else:
    print("WARNING - some rows were lost!")

# Check how many missing values exist after the merge (just to know).
missing_total = data.isna().sum().sum()
print("Total missing values in the merged table:", missing_total)
print()


# ---- Step 4: save the merged dataset ---------------------------------------

# Save the result so that the next tasks can use it without redoing the join.
output_path = "../outputs/merged_dataset.csv"
data.to_csv(output_path, index=False)
print("Merged dataset saved to:", output_path)
