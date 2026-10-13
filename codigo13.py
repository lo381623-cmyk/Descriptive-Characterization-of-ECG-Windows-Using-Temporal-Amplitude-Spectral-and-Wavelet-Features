
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Cargar la matriz de características
df = pd.read_csv("results/features/Final_Feature_Matrix.csv")

# Seleccionar únicamente las columnas numéricas
X = df.select_dtypes(include=["number"])

# Excluir características constantes por la normalización Z-score
excluded_features = ["Mean", "Std", "Variance", "RMS", "Energy"]

X = X.drop(
    columns=excluded_features,
    errors="ignore"
)

# Calcular la matriz de correlación
corr = X.corr()

# Generar y guardar el mapa de calor
os.makedirs("results/figures", exist_ok=True)

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3,
    cbar_kws={"label": "Pearson Correlation"}
)

plt.title("Feature Correlation Matrix", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/figures/Figure_12_Correlation_Heatmap.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Heatmap regenerated without constant features.")
print("Remaining features:", len(X.columns))
print("Excluded features:",
      [col for col in excluded_features if col in df.columns])
