# Task 5 - Prediction Models

## Goal

Use supervised learning to predict the variable `weight_change_kg_6m`
(the change in the patient's weight after 6 months on the diet
program) from the patient, diet and nutritionist features.

This is a **regression** problem because the target is a continuous
number (kilos).

## Method

### 1. Choosing features and target

- **Target**: `weight_change_kg_6m`
- **Features**: all columns except the target itself, the IDs
  (`program_id`, `patient_id`, `nutritionist_id`, `diet_id`) and the
  `record_created_at` date. The IDs are just labels with no information
  for prediction.

### 2. Encoding categorical variables

Models in scikit-learn cannot work with text directly, so I converted
each categorical column (sex, diet_type, diet_name, approach,
specialty) into 0/1 columns using `pd.get_dummies`. This is called
**one-hot encoding**. For example, the column `sex` becomes one column
`sex_m` (1 if male, 0 if female).

After encoding, the dataset has 35 features.

### 3. Train / test split

I split the data into 80% training and 20% test using
`train_test_split` with `random_state=42`. The model learns from the
training set and we evaluate it on the test set, which it has never
seen.

- Training rows: 1896
- Test rows: 474

### 4. Scaling

I scaled the features with `StandardScaler` because models like Linear
Regression and KNN are sensitive to the scale. Tree-based models
(Decision Tree, Random Forest) do not need it but are not hurt by it.

### 5. Models tried

I tried four different models, going from simple to more complex:

| # | Model | Idea |
|---|---|---|
| 1 | **Linear Regression** | Simple baseline. Fits a straight line through the data. Good when the relation is mostly linear. |
| 2 | **KNN (k=5)** | Predicts the average target of the 5 most similar patients. |
| 3 | **Decision Tree** | Builds a tree of yes/no questions about the features. Easy to read but can overfit. |
| 4 | **Random Forest** | An average of many random decision trees. Usually the best of the simple models. |

### 6. Evaluation metrics

For regression I used three metrics:

- **MAE** (Mean Absolute Error): the average error in kg. Lower is better.
- **RMSE** (Root Mean Squared Error): like MAE but punishes big errors more.
- **R²** (R-squared): how much of the variance the model explains.
  1 = perfect, 0 = no better than predicting the mean.

### 7. Hyperparameter tuning

For the best model (Random Forest) I used `GridSearchCV` to try a few
combinations of `n_estimators` (number of trees), `max_depth` (depth
of each tree) and `min_samples_split`, and pick the best one based on
3-fold cross-validation.

## Results - model comparison

| Model | MAE (kg) | RMSE (kg) | R² |
|---|---:|---:|---:|
| Linear Regression | 1.94 | 2.49 | 0.886 |
| KNN (k=5) | 3.63 | 4.54 | 0.619 |
| Decision Tree | 2.49 | 3.24 | 0.805 |
| Random Forest | 1.62 | 2.24 | 0.907 |
| **Random Forest (tuned)** | **1.60** | **2.23** | **0.908** |

The best model is **Random Forest (tuned)**. With an R² of 0.908 it
explains about 91% of the variance in the weight change. The mean
error is only 1.6 kg, which is small considering that the typical
weight change is around 22 kg.

The best parameters found by GridSearch were:

```
n_estimators = 300
max_depth = None
min_samples_split = 2
```

## Most important features

The Random Forest gives a score for each feature based on how much
it was used in the trees. The plot
`outputs/plots/feature_importance.png` shows the top 15.

Top features:

| Feature | Importance |
|---|---:|
| `mean_adherence_pct` | 0.44 |
| `baseline_weight_kg` | 0.22 |
| `age` | 0.12 |
| `sex_m` (male) | 0.08 |
| `years_experience` | 0.03 |
| `height_cm` | 0.02 |

The picture is very clear:

- **Adherence is by far the most important feature.** Almost half of
  the model's decisions are based on how well the patient followed
  the diet.
- **Initial weight** is the second most important feature. Heavier
  patients have more weight to lose.
- **Age** and **sex** also play a role.
- The diet name and the nutritionist approach (which were one-hot
  encoded into many columns) have lower individual importance, but
  their total contribution is still relevant.

## Predicted vs actual

The plot `outputs/plots/predicted_vs_actual.png` shows the predictions
of the Random Forest against the real values. The points are mostly
close to the diagonal (red dashed line), which means the predictions
are accurate.

## Files produced

- `outputs/model_comparison.csv` - table with the metrics of every model
- `outputs/plots/feature_importance.png` - top 15 features
- `outputs/plots/predicted_vs_actual.png` - prediction quality plot

## Conclusions

- Random Forest is the best model with R² = 0.908.
- KNN is the worst model: it has trouble because we have 35 features
  and high-dimensional distances are not very informative.
- Linear Regression is already very good (R² = 0.886), which means
  the relations in the data are mostly linear. The Random Forest
  gives a small but consistent improvement.
- The most useful features for prediction are **adherence**, **initial
  weight**, **age** and **sex**.

## How to run

From the `src` folder:

```
python3 task5_prediction.py
```
