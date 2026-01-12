# ============================================================
# FILE: knn_from_scratch_iris.py
# Objective: Implement K-Nearest Neighbors (KNN) from scratch
#            Evaluate effect of K on classification
# Dataset: Iris (numeric features + discrete class labels)
# ============================================================

# -------------------------------
# STEP 0: Import libraries
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
iris = load_iris()
X = iris.data        # Feature matrix (numeric)
y = iris.target      # Class labels (discrete)

# -------------------------------
# STEP 2: Manual Normalization (From Scratch)
# -------------------------------
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# -------------------------------
# STEP 3: Split Dataset (70% Train, 15% Validation, 15% Test)
# -------------------------------
# First split: Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=22101234,
    shuffle=True
)

# Second split: Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=22101234,
    shuffle=True
)

# -------------------------------
# STEP 4: Define Euclidean Distance
# -------------------------------
def euclidean_distance(x1, x2):
    """Compute Euclidean distance between two vectors"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

# -------------------------------
# STEP 5: Predict single test sample
# -------------------------------
def knn_predict_single(x_test, X_train, y_train, k):
    """
    Predict class of a single test sample using KNN
    Steps:
    a) Compute distances to all training samples
    b) Sort distances ascending
    c) Select K nearest neighbors
    d) Majority voting
    """
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(x_test, X_train[i])
        distances.append((dist, y_train[i]))
    
    # Sort distances
    distances.sort(key=lambda x: x[0])
    
    # Select K nearest neighbors
    k_neighbors = distances[:k]
    
    # Majority voting
    labels = [label for _, label in k_neighbors]
    predicted_class = Counter(labels).most_common(1)[0][0]
    return predicted_class

# -------------------------------
# STEP 6: Predict all test samples
# -------------------------------
def knn_predict(X_test, X_train, y_train, k):
    """Predict all samples in X_test"""
    return np.array([knn_predict_single(x, X_train, y_train, k) for x in X_test])

# -------------------------------
# STEP 7: Accuracy Calculation
# -------------------------------
def accuracy(y_true, y_pred):
    """Compute classification accuracy"""
    return np.sum(y_true == y_pred) / len(y_true)

# -------------------------------
# STEP 8: Evaluate different K values using VALIDATION set
# -------------------------------
k_values = [1, 3, 5, 7]
accuracies = []

for k in k_values:
    y_val_pred = knn_predict(X_val, X_train, y_train, k)
    acc = accuracy(y_val, y_val_pred)
    accuracies.append(acc)
    print(f"K = {k}, Validation Accuracy = {acc:.4f}")

# -------------------------------
# STEP 9: Select best K based on Validation
# -------------------------------
best_k = k_values[np.argmax(accuracies)]
print("Best K based on validation:", best_k)

# -------------------------------
# STEP 10: Final Test Accuracy
# -------------------------------
y_test_pred = knn_predict(X_test, X_train, y_train, best_k)
test_acc = accuracy(y_test, y_test_pred)
print("Final Test Accuracy:", test_acc)

# -------------------------------
# STEP 11: Plot K vs Validation Accuracy
# -------------------------------
plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Validation Accuracy")
plt.title("KNN Validation Accuracy vs K")
plt.grid(True)
plt.show()

# ============================================================
# ALTERNATIVES / VIVA DISCUSSION POINTS
# ============================================================

"""
1. Distance Metric:
   - Currently using Euclidean distance
   - Alternative: Manhattan distance, Minkowski distance

2. K Selection:
   - We used validation set to pick best K
   - Alternative: Cross-validation (k-fold)
   - Could also experiment with weighted voting (closer neighbors count more)

3. Normalization:
   - Done manually
   - Alternative: Standardization (Z-score)
   - Using sklearn's MinMaxScaler or StandardScaler

4. Dataset:
   - Iris is small & clean → high accuracy
   - Alternative: Use datasets with more features or overlap for realistic variation
   - Examples: Wine dataset, Breast Cancer dataset

5. Efficiency:
   - Current code is simple but slow for large datasets
   - Alternative: Use KD-tree, Ball-tree, or sklearn's KNeighborsClassifier
   - Can vectorize distance computation using NumPy for speed

6. Handling Ties in Voting:
   - Currently picks the first most common
   - Alternative: Random tie-break or distance-weighted voting

7. Test Accuracy vs Validation Accuracy:
   - Validation accuracy used to select K
   - Test accuracy is final unbiased performance measure
"""

