
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("results/features/Final_Feature_Matrix.csv")

y = df["Class"].copy()
X = df.drop(columns=["Class"]).select_dtypes(include=["number"]).copy()

# Check for invalid values
X = X.replace([np.inf, -np.inf], np.nan)
valid_rows = X.notna().all(axis=1)

print("Rows excluded due to missing/invalid values:",
      int((~valid_rows).sum()))

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# ==========================================================
# REMOVE NORMALIZATION-INDUCED FEATURES
# ==========================================================

def normalize_name(name):
    return str(name).strip().lower().replace("_", " ")

normalization_features = {
    "mean",
    "std",
    "standard deviation",
    "variance",
    "rms",
    "energy"
}

excluded_by_name = [
    col for col in X.columns
    if normalize_name(col) in normalization_features
]

constant_cols = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

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
    raise ValueError("Fewer than two valid features remain for t-SNE.")

if len(X) <= 30:
    raise ValueError(
        "The current perplexity of 30 requires more than 30 samples."
    )

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite values remain in the feature matrix.")

# ==========================================================
# STANDARDIZATION
# ==========================================================

X_scaled = StandardScaler().fit_transform(X)

# ==========================================================
# t-SNE
# ==========================================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8, 6))

for cls in y.unique():
    idx = y.to_numpy() == cls
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        label=cls,
        s=10
    )

plt.title("t-SNE Projection of ECG Feature Space")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/figures/Figure_11_tSNE.pdf",
    dpi=300
)

plt.close()

print("t-SNE recalculated successfully.")
