import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import make_blobs

# ======================
# 1. Generate Sample Data
# ======================
np.random.seed(42)
X, _ = make_blobs(n_samples=100, n_features=3, centers=3, cluster_std=1.5)
print("Data shape:", X.shape)

# ======================
# 2. PCA from Scratch
# ======================
class PCA_scratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store components and explained variance
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Fit and transform
pca_scratch = PCA_scratch(n_components=2)
X_pca_scratch = pca_scratch.fit_transform(X)

# ======================
# 3. Compare with scikit-learn
# ======================
pca_sklearn = SklearnPCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X)

# Check if results match (up to sign flips)
print("\nMax absolute difference:", np.max(np.abs(X_pca_scratch) - np.abs(X_pca_sklearn)))

# ======================
# 4. Visualization
# ======================
plt.figure(figsize=(12, 5))

# Original 3D Data (first 2 dims for visualization)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Original Data (First 2 Dimensions)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# PCA Projection
plt.subplot(1, 2, 2)
plt.scatter(X_pca_scratch[:, 0], X_pca_scratch[:, 1], alpha=0.7)
plt.title("PCA Projection (Scratch Implementation)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

# ======================
# 5. Explained Variance
# ======================
print("\nExplained Variance Ratio (Scratch):", pca_scratch.explained_variance_ratio_)
print("Explained Variance Ratio (scikit-learn):", pca_sklearn.explained_variance_ratio_)

# Cumulative variance plot
plt.figure()
plt.plot(np.cumsum(pca_scratch.explained_variance_ratio_), 'o-', label='Scratch')
plt.plot(np.cumsum(pca_sklearn.explained_variance_ratio_), 's--', label='scikit-learn')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance")
plt.legend()
plt.grid()
plt.show()