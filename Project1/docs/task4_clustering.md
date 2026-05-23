# Task 4 - Pattern Identification (Clustering)

## Goal

Use unsupervised learning to find natural groups of patients in the data.
The idea is: without telling the algorithm anything about the result of
the diet, can we find groups of patients that look similar? If yes, we
can describe each group and see if some groups are linked to better or
worse diet results.

## Method

I used the **K-Means** algorithm. K-Means works like this:

1. Pick a number of clusters `k`.
2. Place `k` random center points in the data.
3. Assign each patient to the nearest center.
4. Move each center to the average of its assigned patients.
5. Repeat steps 3-4 until the centers stop moving.

The result is `k` groups. Patients inside the same group are more
similar to each other than to patients in other groups.

### Features used

I used 8 numeric features that describe the patient and the program:

```
age, height_cm, baseline_weight_kg, baseline_bmi, sleep_hours,
motivation_score, mean_adherence_pct, years_experience
```

I did **not** use the target `weight_change_kg_6m` because clustering
should not look at the answer; the goal is to find groups based on
the inputs only.

### Scaling

K-Means uses distances. If one feature has big values (like height in
cm: 170) and another has small values (like motivation: 0.7), the big
one dominates the distance and the result is wrong. So I used
`StandardScaler` to put all features on the same scale (mean 0,
standard deviation 1).

### Choosing `k` (elbow method)

I tried `k = 2, 3, ..., 10` and recorded the *inertia* (sum of squared
distances of points to their cluster center). The plot
`outputs/plots/clustering_elbow.png` shows the inertia going down as
`k` grows. The "elbow" is the point where the curve starts to flatten,
because adding more clusters does not improve the result much.

Looking at my elbow plot, the curve starts to flatten around **k = 4**,
so I chose 4 clusters.

## Results - the four clusters

After running K-Means with `k = 4`, I computed the mean of the original
features per cluster to understand each group.

| Feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---:|---:|---:|---:|
| Number of patients | 469 | 545 | 589 | 767 |
| Mean age | 46.3 | 44.5 | 46.5 | 40.4 |
| Mean height (cm) | 168 | **178** | 162 | 167 |
| Mean weight (kg) | 72.4 | **85.3** | 80.8 | 65.9 |
| Mean BMI | 25.6 | 26.9 | **30.8** | 23.6 |
| Mean motivation | **0.49** | 0.75 | 0.78 | 0.78 |
| Mean adherence (%) | **57.7** | 81.7 | 83.4 | 84.6 |
| Sex (F / M) | 266 / 203 | 138 / 407 | 445 / 144 | 433 / 334 |

### Description of each cluster

**Cluster 0 - Low-motivation patients (469 patients)**

This group has the lowest motivation (0.49) and the lowest adherence
(57.7%). The age and BMI are average. Both sexes are represented.
This cluster looks like the patients who do not really follow the
diet program.

**Cluster 1 - Tall and heavy patients, mostly men (545 patients)**

The tallest (178 cm) and heaviest (85 kg) patients. About 75% of this
cluster are men. Motivation and adherence are good.

**Cluster 2 - Shorter patients with high BMI, mostly women (589 patients)**

The shortest (162 cm) patients with the highest BMI (30.8 - already
in the obesity range). About 75% are women. Adherence is good.

**Cluster 3 - Young patients with low BMI and high adherence (767 patients)**

The youngest group (40 years), with the lowest weight (66 kg) and
lowest BMI (23.6 - normal weight). They have the highest adherence
(84.6%). Both sexes are present.

## Mean weight change per cluster

After the clustering I checked the average of the target per cluster
(this was just for interpretation, the target was not used in the
clustering itself):

| Cluster | Mean weight change (kg) |
|---|---:|
| Cluster 0 (low motivation) | **-13.75** (worst result) |
| Cluster 1 (tall/heavy men) | **-27.41** (best result) |
| Cluster 2 (high BMI women) | -23.70 |
| Cluster 3 (young, low BMI) | -21.56 |

This is interesting:

- The patients with **low motivation/adherence** lose much less weight,
  which confirms what we saw in the EDA.
- The **tall, heavy patients** lose more weight than the others. This
  is partly because they have more weight to lose, and partly because
  the men in this cluster show good adherence.
- The young, low-BMI cluster has the best adherence but loses less
  weight in absolute kilos, simply because they were already lighter.

## Plots produced

- `outputs/plots/clustering_elbow.png` - elbow method to choose k
- `outputs/plots/clustering_pca.png` - 2D view of the clusters using PCA

## Files produced

- `outputs/cluster_profiles.csv` - mean of each feature per cluster
- `outputs/clustered_dataset.csv` - the cleaned dataset with one extra
  column `cluster` indicating the cluster of each row

## Conclusions for the next tasks

- The clusters confirm that **adherence and motivation** are key for
  weight loss success.
- The **sex of the patient** is important: men and women form quite
  different profiles (cluster 1 mostly male, cluster 2 mostly female).
- Patients with **low motivation form their own group** with much
  worse results. In a real-world scenario, the nutritionist could try
  to detect these patients early and use a different approach.

## How to run

From the `src` folder:

```
python3 task4_clustering.py
```
