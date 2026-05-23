# Task 5 - Prediction Models
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we use SUPERVISED learning to predict the variable
# weight_change_kg_6m (how many kilos the patient lost or gained
# after 6 months on the diet program).
#
# We try several models, compare them, and pick the best one. We
# also look at which features the best model thinks are important.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---- Step 1: load the cleaned dataset --------------------------------------

data = pd.read_csv("../outputs/cleaned_dataset.csv")
print("Dataset shape:", data.shape)


# ---- Step 2: choose features and target ------------------------------------

# Target: weight_change_kg_6m (the value we want to predict)
# Features: everything that describes the patient, the diet and the
# nutritionist. We drop:
#   - the IDs (they are just labels, no information for prediction)
#   - the target itself
#   - the date column (we are not doing time analysis)

target_col = "weight_change_kg_6m"
columns_to_drop = [
    target_col,
    "program_id",
    "patient_id",
    "nutritionist_id",
    "diet_id",
    "record_created_at",
]
X = data.drop(columns=columns_to_drop)
y = data[target_col]

print("Number of features (before encoding):", X.shape[1])


# ---- Step 3: encode categorical features -----------------------------------

# Models from scikit-learn need numbers, not text. So I convert each
# categorical column into several 0/1 columns using one-hot encoding.
# pd.get_dummies does this automatically.

X = pd.get_dummies(X, drop_first=True)
print("Number of features (after encoding):", X.shape[1])


# ---- Step 4: split the data into training and test sets --------------------

# We train the models on the training set and we test them on data
# they have never seen. random_state=42 makes the split reproducible.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training rows:", X_train.shape[0])
print("Test rows:", X_test.shape[0])


# ---- Step 5: scale the features --------------------------------------------

# Some models (Linear Regression, KNN) work better when the features
# are scaled. Tree-based models (Decision Tree, Random Forest) do not
# need scaling but it does not hurt them.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---- Step 6: train and evaluate several models -----------------------------

# A small helper to print the metrics of a model.
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:25s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


print("\n--- Model comparison ---")
results = []

# Model 1: Linear Regression - simple baseline
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
results.append(evaluate("Linear Regression", y_test, lr.predict(X_test_scaled)))

# Model 2: K-Nearest Neighbors (KNN)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
results.append(evaluate("KNN (k=5)", y_test, knn.predict(X_test_scaled)))

# Model 3: Decision Tree
dt = DecisionTreeRegressor(random_state=42, max_depth=8)
dt.fit(X_train, y_train)
results.append(evaluate("Decision Tree", y_test, dt.predict(X_test)))

# Model 4: Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))


# ---- Step 7: tune the best model with GridSearchCV -------------------------

# Random Forest is usually the best one. We try a small grid of values
# for the main hyperparameters and pick the combination that gives the
# best result on cross-validation.

print("\n--- Tuning Random Forest with GridSearchCV ---")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)

best_rf = grid.best_estimator_
results.append(evaluate("Random Forest (tuned)", y_test, best_rf.predict(X_test)))


# ---- Step 8: save the comparison table -------------------------------------

results_df = pd.DataFrame(results).round(3)
results_df.to_csv("../outputs/model_comparison.csv", index=False)
print("\nSaved: ../outputs/model_comparison.csv")
print(results_df)


# ---- Step 9: feature importance of the best model --------------------------

# Random Forest gives a score for each feature: how useful it was for
# the prediction. We plot the top 15.

importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 5))
top[::-1].plot.barh(color="steelblue")
plt.title("Top 15 most important features (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../outputs/plots/feature_importance.png", dpi=120)
plt.close()
print("\nSaved: feature_importance.png")
print("\nTop 10 most important features:")
print(top.head(10).round(4))


# ---- Step 10: predicted vs actual plot -------------------------------------

# Visualize how good the predictions are by plotting predicted values
# against the real values. A perfect model would have all the points
# on the diagonal.

y_pred = best_rf.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], "r--", label="Perfect prediction")
plt.xlabel("Actual weight change (kg)")
plt.ylabel("Predicted weight change (kg)")
plt.title("Predicted vs actual - Random Forest (tuned)")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/plots/predicted_vs_actual.png", dpi=120)
plt.close()
print("Saved: predicted_vs_actual.png")

print("\nDone.")
