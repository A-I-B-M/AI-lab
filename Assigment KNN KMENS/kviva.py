# ============================================================
# K-MEANS CLUSTERING FROM SCRATCH
# Implemented manually without sklearn or pandas
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 0: Set random seed (Student ID)
# WHY:
# - To make random initialization reproducible
# VIVA ALTERNATIVE:
# - "Seed ensures same result every run"
# ------------------------------------------------------------
np.random.seed(20231234)   # <-- replace with YOUR student ID

# ------------------------------------------------------------
# STEP 1: Load dataset
# WHAT I DID:
# - Loaded 2D points from dataset.txt
# VIVA ALTERNATIVE:
# - "Each row represents one data point"
# ------------------------------------------------------------
data = np.loadtxt("dataset.txt")

# ------------------------------------------------------------
# STEP 2: Normalize features
# WHAT I DID:
# - Used Z-score normalization
# WHY:
# - Prevents one feature dominating distance
# VIVA ALTERNATIVE:
# - "Min-Max normalization could also be used"
# ------------------------------------------------------------
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
Data = (data - mean) / std

# ------------------------------------------------------------
# STEP 3: Define K-Means from scratch
# ------------------------------------------------------------
def kmeans_from_scratch(Data, K, max_iters=100, tol=1e-4):

    # --------------------------------------------------------
    # STEP 3.1: Initialize K cluster centers randomly
    # WHAT I DID:
    # - Selected K random points as initial centers
    # VIVA ALTERNATIVE:
    # - "K-means++ can be used for better initialization"
    # --------------------------------------------------------
    centers = Data[np.random.choice(len(Data), K, replace=False)]

    # --------------------------------------------------------
    # STEP 4 & 5: Iterative optimization
    # --------------------------------------------------------
    for _ in range(max_iters):

        # STEP 4: Create empty clusters
        # VIVA:
        # - "Each list stores points for one cluster"
        clusters = [[] for _ in range(K)]

        # STEP 5a: Assign points to nearest center
        # WHAT I DID:
        # - Used Euclidean distance
        # VIVA ALTERNATIVE:
        # - "Manhattan distance can also be used"
        for point in Data:
            distances = [np.linalg.norm(point - c) for c in centers]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # STEP 5b: Update cluster centers
        # WHAT I DID:
        # - Computed mean of each cluster
        # VIVA:
        # - "Mean minimizes squared error"
        new_centers = []
        for i in range(K):
            if clusters[i]:
                new_centers.append(np.mean(clusters[i], axis=0))
            else:
                new_centers.append(centers[i])  # empty cluster case

        new_centers = np.array(new_centers)

        # STEP 5c: Convergence check
        # WHAT I DID:
        # - Compared movement of centers
        # VIVA ALTERNATIVE:
        # - "Stopping after fixed iterations is also possible"
        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    # --------------------------------------------------------
    # STEP 6: Compute inertia
    # WHAT I DID:
    # - Sum of squared distances from points to center
    # VIVA:
    # - "Lower inertia means tighter clusters"
    # --------------------------------------------------------
    inertia = 0
    for i in range(K):
        for point in clusters[i]:
            inertia += np.sum((point - centers[i]) ** 2)

    return clusters, centers, inertia

# ------------------------------------------------------------
# STEP 7 & 8: Run for different values of K
# ------------------------------------------------------------
K_values = [2, 4, 6, 7]

for K in K_values:
    clusters, centers, inertia = kmeans_from_scratch(Data, K)

    # Plot clusters
    plt.figure(figsize=(6, 5))
    for cluster in clusters:
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1])

    plt.scatter(centers[:, 0], centers[:, 1],
                marker='X', s=200, c='black')

    plt.title(f"K-Means Clustering (K = {K})")
    plt.xlabel("Feature 1 (Normalized)")
    plt.ylabel("Feature 2 (Normalized)")
    plt.show()

    print(f"K = {K}, Inertia = {inertia:.4f}")
