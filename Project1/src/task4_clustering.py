# Task 4 - Pattern Identification (Clustering)
# Course: Elementos de Inteligencia Artificial e Ciencia de Dados (EIACD) 2025/26
#
# In this task we use UNSUPERVISED learning to find groups (clusters) of
# patients with similar profiles. We do not use the target variable here.
# The idea is to see if there are natural groups in the data and to
# describe them.
#
# We use the K-Means algorithm. To find a good number of clusters we
# use the "elbow method" (we run K-Means with different numbers of
# clusters and pick the value where the curve makes an "elbow").


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ---- Step 1: load the cleaned dataset --------------------------------------

data = pd.read_csv("../outputs/cleaned_dataset.csv")
print("Dataset shape:", data.shape)


# ---- Step 2: select the features used for clustering ----------------------

# I use only numeric features that describe the patient and the program.
# I do NOT include the target weight_change_kg_6m because clustering is
# unsupervised: we want to find groups based on the input features only.

features = [
    "age",
    "height_cm",
    "baseline_weight_kg",
    "baseline_bmi",
    "sleep_hours",
    "motivation_score",
    "mean_adherence_pct",
    "years_experience",
]
X = data[features].copy()
print("Features used for clustering:", features)


# ---- Step 3: scale the features --------------------------------------------

# K-Means uses distances, so all features must be on the same scale.
# Otherwise a feature like height_cm (values around 170) would dominate
# a feature like motivation_score (values around 0.7).

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---- Step 4: find a good number of clusters (elbow method) ----------------

# We run K-Means for k = 2, 3, ..., 10 and we record the "inertia"
# (the sum of squared distances of the points to their cluster center).
# The smaller the inertia, the closer the points are to their center.
# When the curve makes an "elbow" we pick that k.

inertias = []
k_values = list(range(2, 11))
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k}, inertia={km.inertia_:.0f}")

plt.figure(figsize=(7, 4))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow method - choose the k where the line bends")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/plots/clustering_elbow.png", dpi=120)
plt.close()
print("Saved: clustering_elbow.png")


# ---- Step 5: run K-Means with the chosen k ---------------------------------

# Looking at the elbow plot I picked k = 4 (the curve starts to flatten
# after this point).
chosen_k = 4
print(f"\nUsing k = {chosen_k}")

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Add the cluster label to the dataset.
data["cluster"] = labels


# ---- Step 6: visualize the clusters in 2D using PCA ------------------------

# PCA reduces the data to 2 dimensions so we can plot it. We lose some
# information but it gives a visual idea of the clusters.

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
for c in range(chosen_k):
    mask = labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c}", alpha=0.6)
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title(f"Clusters projected on 2D (PCA), k = {chosen_k}")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/plots/clustering_pca.png", dpi=120)
plt.close()
print("Saved: clustering_pca.png")


# ---- Step 7: describe each cluster -----------------------------------------

# To interpret the clusters we compute the mean of the original (not
# scaled) features per cluster. This gives a profile of each group.

print("\n--- Cluster sizes ---")
print(data["cluster"].value_counts().sort_index())

print("\n--- Mean of features by cluster ---")
profile = data.groupby("cluster")[features].mean().round(2)
print(profile)

# It is also interesting to see the average weight change per cluster,
# even if we did not use it for the clustering itself.
print("\n--- Mean weight change by cluster (not used in clustering) ---")
print(data.groupby("cluster")["weight_change_kg_6m"].mean().round(2))

# Save the profile table for the report.
profile.to_csv("../outputs/cluster_profiles.csv")
print("\nSaved: ../outputs/cluster_profiles.csv")


# ---- Step 8: save the dataset with cluster labels --------------------------

output_path = "../outputs/clustered_dataset.csv"
data.to_csv(output_path, index=False)
print("Clustered dataset saved to:", output_path)
