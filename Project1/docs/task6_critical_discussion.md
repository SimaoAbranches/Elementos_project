# Task 6 - Critical Discussion of Results

## Goal

Look back at all the work done in Tasks 1 to 5 and discuss the results.
The point here is not to describe what each model did again, but to
**explain why** the results came out like that, what the limits of the
work are, and what could be done better.

## 1. Model comparison and why one is better than another

| Model | R² | MAE (kg) | Why this result? |
|---|---:|---:|---|
| Linear Regression | 0.886 | 1.94 | Already very good. The data has a strong linear relation between adherence/weight/age and the target, which is exactly what this model assumes. |
| KNN (k=5) | 0.619 | 3.63 | The worst. With 35 features after one-hot encoding, distances become less meaningful (the "curse of dimensionality"). Many features are 0/1 dummy variables, which mess up the distances. |
| Decision Tree | 0.805 | 2.49 | A single tree is unstable and tends to overfit. Limiting `max_depth` to 8 helped but it still cannot capture all the structure. |
| **Random Forest (tuned)** | **0.908** | **1.60** | The best. By averaging 300 trees, the random forest reduces the variance of a single tree and captures both linear and non-linear effects. |

The fact that **Linear Regression is so close to Random Forest** is a
useful observation. It tells me that the relation between the inputs
and the target is mostly linear. Random Forest only adds a small extra
gain by capturing some non-linear interactions.

## 2. Why did some models do badly?

- **KNN failed** because we have 35 features and most of them are
  one-hot dummy variables. KNN works on distances, and high-dimensional
  distances do not separate similar from different patients well.
- **The single Decision Tree overfits**: it learns the training data
  too closely and does not generalize. The Random Forest fixes this
  by combining many trees with random subsets of the data.

## 3. Which features really matter?

Both the EDA and the Random Forest agree on the same answer:

1. `mean_adherence_pct` (44% importance)
2. `baseline_weight_kg` (22%)
3. `age` (12%)
4. `sex_m` (8%)

In other words: if a patient follows the diet, weighs more at the
start, is younger, and is male, they will likely lose more weight.

The features that I expected to be more important but turned out
not to be:

- `sleep_hours` was almost useless (importance < 1% and correlation
  near 0). Sleep quality probably matters in real life, but in this
  dataset it does not seem to be linked with the result.
- The `specialty` of the nutritionist (`diabetes`, `general`,
  `sports`, `weight_loss`) showed almost no difference between
  groups. The specific diet matters more than the nutritionist's
  field of specialty.

## 4. Practical questions answered

The project description asked questions like "How much weight will I
lose if I follow the Keto diet?" or "Which type of nutritionist
maximises my weight loss?". Based on the model and the EDA we can
give the following answers:

- **Best diet**: low-carb diets like Atkins Phase2 and LowCarb Classic
  give the biggest average weight loss (-24 kg vs -20 kg for the
  Mediterranean).
- **Best nutritionist approach**: a `strict` approach gives the best
  average results (-24 kg), `data_driven` gives the smallest (-20 kg).
- **The patient himself matters more than the nutritionist**: the
  three strongest features (adherence, initial weight, age) are all
  about the patient. The nutritionist features (approach, experience,
  specialty) have a smaller effect.

## 5. Limitations of this work

Even with R² = 0.908, the model is far from perfect. There are several
reasons:

1. **The dataset has noise**. Some columns had impossible values
   (height of 10 cm, age of 0, BMI of 9410). I cleaned the obvious
   cases but smaller errors probably remain.
2. **Missing values were filled with the median/mode**. This is a
   simple choice, but it can introduce a small bias because all the
   filled rows get the same value.
3. **Adherence is measured *during* the diet**, not before. So in
   practice we cannot use it to predict the result before the diet
   even starts. A more useful (but harder) model would predict
   adherence first, and then the weight change.
4. **The dataset only describes 6 months**. We have no information
   about what happens after.
5. **The target was clipped to ±50 kg** during cleaning. This avoided
   wild outliers but it may have removed some real extreme cases.

## 6. What I would do differently

If I had more time I would try:

- **More feature engineering**: for example, compute the BMI category
  (under, normal, over, obese) instead of using the raw BMI, or
  compute the difference between sodium target and recommended.
- **Try gradient boosting** (XGBoost or LightGBM). These are usually
  even better than Random Forest, but they need more careful tuning.
- **Predict success / failure** as a classification problem. For
  example, define "success" as losing more than 5% of the body
  weight, and try to predict yes/no. This might give a more
  practical answer.
- **Cross-validate the final model** with several different splits to
  make sure the result is stable.

## 7. What I learned

- Cleaning the data is the longest and probably most important step.
  I spent a lot of time finding duplicated columns, fixing impossible
  values, and standardizing categorical text.
- The simplest model (Linear Regression) was almost as good as the
  best one. This shows that picking the most complex model is not
  always the right answer.
- The most important variables were not surprising (adherence and
  initial weight), but seeing it in the data and confirmed by the
  model was a good lesson.
- Looking at the data with plots helped me understand the problem
  much better than just looking at numbers.

## Conclusion

Out of four models tested, the tuned Random Forest gave the best
results (R² = 0.908, MAE = 1.6 kg). The most important features for
predicting weight change are how well the patient follows the diet,
their starting weight, their age and their sex. The nutritionist's
approach matters too, but less than the patient's own behaviour.

Even with a strong model, real life has many factors we did not
capture. The model should be used as a tool to guide the
nutritionist, not as a final answer.
