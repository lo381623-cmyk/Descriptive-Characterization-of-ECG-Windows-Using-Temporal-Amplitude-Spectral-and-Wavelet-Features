import pandas as pd
import numpy as np

df = pd.read_csv("results/features/FFT_Features.csv")

cols = ["Dominant_Frequency", "Spectral_Centroid", "Bandwidth", "Spectral_Entropy"]
classes = ["Normal", "SVEB", "VEB"]

# Decimales por característica
dec = {"Dominant_Frequency": 2, "Spectral_Centroid": 2,
       "Bandwidth": 2, "Spectral_Entropy": 3}

# --- Tabla 3: media ± desviación estándar por clase ---
rows = []
for c in cols:
    row = {"Feature": c}
    for k in classes:
        v = df.loc[df["Class"] == k, c]
        row[k] = f"{v.mean():.{dec[c]}f} ± {v.std():.{dec[c]}f}"
    rows.append(row)

table3 = pd.DataFrame(rows)
print(table3.to_string(index=False))
table3.to_csv("results/tables/Table_3_Spectral_Features.csv", index=False)

# --- Chequeos útiles para redactar el párrafo ---
print("\nN por clase:")
print(df["Class"].value_counts())

print("\nMediana [Q1, Q3] de entropía espectral:")
for k in classes:
    v = df.loc[df["Class"] == k, "Spectral_Entropy"]
    print(f"{k}: {v.median():.3f} [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]")

print("\nEntropía máxima teórica (nats), espectro plano de 251 bins:",
      round(np.log(251), 3))