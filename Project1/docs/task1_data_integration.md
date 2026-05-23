# Task 1 - Data Integration

## Goal

Combine the four CSV files into a single dataset that can be used in the
following tasks. This is the first step of the project and it makes sure
that every piece of information about a patient, a diet, a nutritionist
and the result of the diet program is on the same row.

## Input files

The data folder has four CSV files:

| File | Rows | Columns | What it contains |
|------|-----:|--------:|------------------|
| patients.csv | 1000 | 11 | Personal information of each patient (sex, age, weight, BMI, etc.) |
| diets.csv | 10 | 9 | Description of each diet (type, % of carbs, protein, fat, etc.) |
| nutritionists.csv | 20 | 5 | Information about each nutritionist (approach, experience, specialty) |
| outcomes.csv | 2523 | 9 | Result of each diet program (which patient, which diet, which nutritionist, weight change, etc.) |

## Method

The "outcomes" table is the central one because every row already has the
three IDs that link to the other tables:

- `patient_id` -> patients table
- `diet_id` -> diets table
- `nutritionist_id` -> nutritionists table

So the idea is simple: start from `outcomes` and add the columns of the
other three tables one by one. To do this I used the `merge` function
from pandas with `how="left"`. A left merge keeps every row of the
outcomes table even if some ID has no match. This way I can check at
the end if any rows were lost.

Steps in the script `src/task1_integration.py`:

1. Read the four CSV files with `pd.read_csv()`.
2. Print the shape of each table (just to confirm the files were loaded).
3. Merge `outcomes` with `patients` using `patient_id`.
4. Merge the result with `diets` using `diet_id`.
5. Merge again with `nutritionists` using `nutritionist_id`.
6. Check that the number of rows is still equal to the number of rows
   in `outcomes` (no rows lost).
7. Save the final table to `outputs/merged_dataset.csv`.

## Result

The merged dataset has:

- **2523 rows** (one per diet program, same as the outcomes table)
- **31 columns** (combination of all the four tables)

No rows were lost during the merge, which means every program in the
outcomes table found a matching patient, diet and nutritionist.

The merged file is saved at `outputs/merged_dataset.csv` and it is the
input for Task 2.

## Things I noticed

While checking the columns I already noticed some things that will be
useful for Task 2:

- There seem to be some columns that look duplicated:
  - `baseline_bmi` and `bmi_redundant` in patients
  - `years_experience` and `experience_years` in nutritionists
  - `mean_adherence_pct` and `adherence_ratio` in outcomes
- The merged table has 2334 missing values in total. These will need
  to be handled in Task 2.
- The column `motivation_score` (from patients) and `motivation_score_program`
  (from outcomes) are different things and should not be confused.

## How to run

From the `src` folder:

```
python3 task1_integration.py
```

The script will print the shapes of the tables and save the merged
dataset to `../outputs/merged_dataset.csv`.
