
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("results/features/Final_Feature_Matrix.csv")

y = df["Class"].copy()
X = df.drop(columns=["Class"]).select_dtypes(include=["number"]).copy()

# Check for invalid values; retain the same rows for X and y
X = X.replace([np.inf, -np.inf], np.nan)
valid_rows = X.notna().all(axis=1)

print("Rows excluded due to missing/invalid values:",
      int((~valid_rows).sum()))

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# ==========================================================
# REMOVE NORMALIZATION-INDUCED FEATURES
# ==========================================================

# Exact feature names, ignoring capitalization and underscores
normalization_features = {
    "mean",
    "std",
    "standard deviation",
    "variance",
    "rms",
    "energy"
}

def normalize_name(name):
    return str(name).strip().lower().replace("_", " ")

excluded_by_name = [
    col for col in X.columns
    if normalize_name(col) in normalization_features
]

# Detect exact constants
constant_cols = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

# Detect numerically near-constant features relative to their scale
near_constant_cols = []

for col in X.columns:
    values = X[col].to_numpy(dtype=float)
    scale = max(np.max(np.abs(values)), 1e-300)

    if np.std(values) / scale < 1e-10:
        near_constant_cols.append(col)

excluded_cols = set(
    excluded_by_name + constant_cols + near_constant_cols
)

X = X.drop(columns=list(excluded_cols), errors="ignore")

print("Excluded features:", sorted(excluded_cols))
print("Features retained:", list(X.columns))

if X.shape[1] < 2:
    raise ValueError("Fewer than two valid features remain for PCA.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite values remain in the feature matrix.")

# ==========================================================
# STANDARDIZATION
# ==========================================================

X_scaled = StandardScaler().fit_transform(X)

# ==========================================================
# PCA
# ==========================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained by two components:",
      pca.explained_variance_ratio_.sum())

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8, 6))

for cls in y.unique():
    idx = y.to_numpy() == cls
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=cls,
        s=10
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of ECG Feature Space")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/figures/Figure_10_PCA.pdf",
    dpi=300
)

plt.close()

print("PCA recalculated successfully.")
