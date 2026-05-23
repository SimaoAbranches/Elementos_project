# Task 3 - Exploratory Data Analysis (EDA)

## Goal

Look at the cleaned dataset to understand the variables, find patterns
and form a first idea of which features can be useful to predict the
target `weight_change_kg_6m` (the kilos a patient lost or gained after
6 months on the diet).

## Input

`outputs/cleaned_dataset.csv` (the result of Task 2) - 2370 rows,
27 columns, no missing values.

## What I did

### 1. Descriptive statistics

I used `data.describe()` for the numeric columns and `value_counts()`
for the categorical ones. This gives a first idea of the typical values
and how many patients are in each category.

### 2. Distribution of the variables

I plotted a histogram for each main numeric variable
(see `outputs/plots/histograms.png`). This shows if the data is
balanced, skewed, or concentrated in a small range.

### 3. Correlation between numeric variables

I computed the correlation matrix and made a heatmap
(`outputs/plots/correlation_heatmap.png`). The correlation is a number
between -1 and 1:

- close to 1 -> the two variables move in the same direction
- close to -1 -> they move in opposite directions
- close to 0 -> no clear linear relation

### 4. Effect of each categorical variable on the target

For each categorical variable I made a boxplot of the target grouped
by the category, and I also computed the mean weight change per group.

## Main findings

### Correlation with the target `weight_change_kg_6m`

| Feature | Correlation |
|---|---:|
| `mean_adherence_pct` | **-0.67** |
| `motivation_score` | **-0.54** |
| `baseline_weight_kg` | -0.47 |
| `age` | +0.32 |
| `height_cm` | -0.30 |
| `years_experience` | -0.25 |
| `baseline_bmi` | -0.24 |
| `sleep_hours` | 0.00 |

The correlation is negative because the target is the *change* in
weight: a more negative number means more kilos lost. So a strong
**negative** correlation here means the variable helps the patient
**lose more** weight.

The two strongest predictors are **adherence** (how well the patient
follows the diet) and **motivation**. This makes sense: the diet only
works if the patient actually follows it.

`sleep_hours` has almost no relation with the target, so it is probably
not useful for prediction.

### Sex of the patient

| Sex | Mean weight change (kg) |
|---|---:|
| Female (f) | -19.29 |
| Male (m) | -24.95 |

Men lose on average about 5 kg more than women in 6 months. This is
a clear difference.

### Diet type and diet name

| Diet type | Mean weight change |
|---|---:|
| balanced | -21.04 |
| high_protein | -21.16 |
| plant | -21.45 |
| low_carb | **-23.79** |

| Diet name (top 3 / bottom 3) | Mean weight change |
|---|---:|
| Atkins Phase2 | **-24.13** |
| LowCarb Classic | **-24.08** |
| SugarFree | -23.18 |
| ... | |
| HighProtein | -20.72 |
| AthletePro | -20.70 |
| Mediterranean | -20.32 |

The low_carb diets (Atkins, LowCarb Classic) give the biggest weight
loss. The Mediterranean diet has the smallest effect.

### Nutritionist approach

| Approach | Mean weight change |
|---|---:|
| strict | **-24.12** |
| motivational | -22.01 |
| empathetic | -21.71 |
| data_driven | -20.31 |

A `strict` approach is associated with bigger weight loss. The
`data_driven` approach has the smallest effect.

### Specialty of the nutritionist

| Specialty | Mean weight change |
|---|---:|
| general | -22.11 |
| sports | -22.00 |
| weight_loss | -21.86 |
| diabetes | -21.61 |

The specialty does not seem to make a big difference (all values are
between -21.6 and -22.1).

### Smoker vs non-smoker

| Smoker | Mean weight change |
|---|---:|
| No | -22.73 |
| Yes | -19.19 |

Non-smokers lose about 3.5 kg more on average.

### Adherence vs weight change

The scatter plot `adherence_vs_weight.png` shows a clear trend:
patients with higher adherence have more weight loss. This matches
the strong correlation of -0.67.

## Plots produced

All plots are saved in `outputs/plots/`:

- `histograms.png` - distribution of the main numeric variables
- `correlation_heatmap.png` - correlation between numeric variables
- `weight_change_by_sex.png`
- `weight_change_by_diet_type.png`
- `weight_change_by_diet_name.png`
- `weight_change_by_approach.png`
- `adherence_vs_weight.png`

## Conclusions for the next tasks

- The most important features for predicting weight change are
  **adherence**, **motivation**, **baseline weight** and **age**.
- The categorical variables that matter most are **sex**, **diet name**
  and **nutritionist approach**.
- `sleep_hours` and `specialty` look weak and may not help the model
  much, but I will still keep them and see what the model says in
  Task 5.

## How to run

From the `src` folder:

```
python3 task3_eda.py
```
