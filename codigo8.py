import os
import numpy as np
import pandas as pd
import pywt

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_DATA = "results/dataset/Preprocessed_ECG_Beats.csv"
LABELS = "results/dataset/Balanced_Labels.csv"

OUTPUT_FEATURES = "results/features"
OUTPUT_TABLES = "results/tables"

os.makedirs(OUTPUT_FEATURES, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

beats = pd.read_csv(INPUT_DATA).values
labels = pd.read_csv(LABELS)["Class"]

print("Number of beats:", len(beats))

# ==========================================================
# WAVELET CONFIG
# ==========================================================

WAVELET = "db4"
LEVEL = 4

features = []

energy_A4 = []
energy_D4 = []
energy_D3 = []
energy_D2 = []
energy_D1 = []

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

for beat in beats:

    coeffs = pywt.wavedec(beat, WAVELET, level=LEVEL)

    # coeffs = [A4, D4, D3, D2, D1]

    A4, D4, D3, D2, D1 = coeffs

    def energy(x):
        return np.sum(np.square(x))

    eA4 = energy(A4)
    eD4 = energy(D4)
    eD3 = energy(D3)
    eD2 = energy(D2)
    eD1 = energy(D1)

    total = eA4 + eD4 + eD3 + eD2 + eD1

    features.append([
        eA4 / total,
        eD4 / total,
        eD3 / total,
        eD2 / total,
        eD1 / total
    ])

    energy_A4.append(eA4)
    energy_D4.append(eD4)
    energy_D3.append(eD3)
    energy_D2.append(eD2)
    energy_D1.append(eD1)

# ==========================================================
# DATAFRAME
# ==========================================================

df = pd.DataFrame(features, columns=[
    "E_A4", "E_D4", "E_D3", "E_D2", "E_D1"
])

df["Class"] = labels

df.to_csv(
    os.path.join(
        OUTPUT_FEATURES,
        "Wavelet_Features.csv"
    ),
    index=False
)

# ==========================================================
# SUMMARY TABLE
# ==========================================================

summary = df.groupby("Class").agg(["mean", "std"])

summary.to_csv(
    os.path.join(
        OUTPUT_TABLES,
        "Table_8_Wavelet_Features.csv"
    )
)

print("Wavelet features extracted successfully.")