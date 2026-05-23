# Task 2 - Data Cleaning and Processing

## Goal

Take the merged dataset from Task 1 and prepare it for analysis. The
data has some duplicated columns, some impossible values and many
missing values. After this task we should have a clean dataset that
the next steps can use without problems.

## Input

`outputs/merged_dataset.csv` (the result of Task 1)

- 2523 rows
- 31 columns
- 2334 missing values in total

## Steps

### Step 1 - Remove duplicated columns

While looking at the merged dataset I found four columns that have
the same information as another column. Keeping both versions does
not help and can confuse the models, so I dropped them.

| Column dropped | Why |
|---|---|
| `bmi_redundant` | Same as `baseline_bmi`, but it also has wrong values like 9410 |
| `experience_years` | Same as `years_experience` |
| `adherence_ratio` | Same as `mean_adherence_pct` divided by 100 |
| `total_macros` | Just the sum of `carb_pct + protein_pct + fat_pct` |

After this step the dataset has **27 columns**.

### Step 2 - Standardize text in categorical columns

Some categorical columns had the same value written in different ways.
For example, the `sex` column had values like "F", " F" (with a space),
"f", "female" and "M", "m", "male". Without fixing this, pandas treats
"F" and " F" as different categories, which is wrong.

What I did:

- `sex`: lowercase, strip spaces, replace "female"->"f" and "male"->"m".
  Final values: only `f` and `m`.
- `approach`: lowercase and strip spaces. The "Strict" and "strict"
  versions were merged into one. Final values: `data_driven`,
  `empathetic`, `motivational`, `strict`.
- `diet_type`, `diet_name`, `specialty`: stripped whitespace just in
  case.

### Step 3 - Fix impossible values (outliers)

Some columns have values that cannot be true in real life. For example,
no one is 10 cm tall or sleeps -1 hours. I defined a realistic range
for each numeric column and replaced the values outside the range
with `NaN`. After that they are treated as missing values and filled
in step 4.

The ranges I used:

| Column | Allowed range | Reason |
|---|---|---|
| `age` | 5 - 100 | A patient cannot be 0 or 200 years old |
| `height_cm` | 100 - 220 | Adult height range |
| `baseline_weight_kg` | 30 - 250 | Realistic weight range |
| `sleep_hours` | 0 - 24 | A day has 24 hours |
| `motivation_score` | 0 - 1 | The column is supposed to be a 0-1 score |
| `years_experience` | 0 - 50 | A nutritionist cannot have 99 years of experience |
| `mean_adherence_pct` | 0 - 100 | It is a percentage |
| `weight_change_kg_6m` | -50 to +50 | Big changes but still realistic in 6 months |

The number of values fixed for each column is printed by the script.

### Step 4 - Drop rows where the target is missing

The variable we want to predict in Task 5 is `weight_change_kg_6m`
(how many kilos the patient lost or gained after 6 months). If this
value is missing we cannot use the row to train a model, so we drop
those rows. **153 rows** were dropped.

### Step 5 - Fill remaining missing values

For the rest of the missing values I used a simple rule:

- **Numeric columns** -> filled with the **median** of the column.
  I chose the median (and not the mean) because it is less affected
  by extreme values.
- **Categorical columns** -> filled with the **mode** (the most
  frequent value).

The script prints which value was used for each column.

## Result

The cleaned dataset has:

- **2370 rows** (we lost 153 rows because the target was missing)
- **27 columns**
- **0 missing values**

The cleaned file is saved at `outputs/cleaned_dataset.csv` and it is
the input for Task 3.

## Things I noticed

- The `years_experience` column had a lot of values equal to 99, which
  was the main outlier source (120 rows). It looks like 99 was used as
  a code for "unknown".
- The `bmi_redundant` column had some values up to 9410, clearly wrong.
- Most missing values came from the patient information (age, sleep
  hours) and from the diet program results (adherence, weight change).

## How to run

From the `src` folder:

```
python3 task2_cleaning.py
```

The script reads `../outputs/merged_dataset.csv` and saves
`../outputs/cleaned_dataset.csv`.
