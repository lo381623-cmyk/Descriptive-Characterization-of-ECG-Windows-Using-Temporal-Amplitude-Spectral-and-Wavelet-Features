import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

SEED = 42

df = pd.read_csv("results/features/Final_Feature_Matrix.csv")
y = df["Class"].copy()
X = df.drop(columns=["Class"]).select_dtypes(include=["number"]).copy()

X = X.replace([np.inf, -np.inf], np.nan)
valid_rows = X.notna().all(axis=1)
print("Rows excluded:", int((~valid_rows).sum()))
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

normalization_features = {
    "mean", "std", "standard deviation", "variance",
    "rms", "energy", "spectral energy"
}

def normalize_name(name):
    return str(name).strip().lower().replace("_", " ")

excluded = [c for c in X.columns if normalize_name(c) in normalization_features]
X = X.drop(columns=excluded)

print("Excluded:", excluded)
print("Retained:", X.shape[1], list(X.columns))   # deben ser 14

# ---------------- Correlación ----------------
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), cmap="coolwarm", vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.5, square=True,
            cbar_kws={"label": "Pearson Correlation"})
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("results/figures/Figure_12_Correlation_Heatmap.pdf")
plt.close()

# ---------------- Estandarización ----------------
X_scaled = StandardScaler().fit_transform(X)

# ---------------- PCA ----------------
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance:", pca.explained_variance_ratio_,
      "Total:", pca.explained_variance_ratio_.sum())

# ---------------- t-SNE ----------------
X_tsne = TSNE(n_components=2, random_state=SEED,
              init="pca", perplexity=30).fit_transform(X_scaled)

# ---------------- Gráficas ----------------
def scatter(Z, xl, yl, title, path):
    plt.figure(figsize=(8, 6))
    for cls in y.unique():
        idx = y.to_numpy() == cls
        plt.scatter(Z[idx, 0], Z[idx, 1], label=cls, s=10)
    plt.xlabel(xl); plt.ylabel(yl); plt.title(title)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

scatter(X_pca, "PC1", "PC2", "PCA of ECG Feature Space",
        "results/figures/Figure_10_PCA.pdf")
scatter(X_tsne, "Dimension 1", "Dimension 2", "t-SNE Projection of ECG Feature Space",
        "results/figures/Figure_11_tSNE.pdf")

print("Done.")