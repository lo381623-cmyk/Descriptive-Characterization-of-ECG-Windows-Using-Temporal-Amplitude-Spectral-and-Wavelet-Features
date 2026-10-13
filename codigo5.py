import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_DATA = "results/dataset/Preprocessed_ECG_Beats.csv"
LABELS = "results/dataset/Balanced_Labels.csv"

OUTPUT_DATA = "results/features"
OUTPUT_TABLES = "results/tables"

os.makedirs(OUTPUT_DATA, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

beats = pd.read_csv(INPUT_DATA).values
labels = pd.read_csv(LABELS)

print("Number of beats:", len(beats))

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

features = []

for beat in beats:

    mean = np.mean(beat)

    std = np.std(beat)

    var = np.var(beat)

    rms = np.sqrt(np.mean(beat**2))

    maximum = np.max(beat)

    minimum = np.min(beat)

    peak_to_peak = np.ptp(beat)

    skewness = skew(beat)

    kurt = kurtosis(beat)

    energy = np.sum(beat**2)

    features.append([
        mean,
        std,
        var,
        rms,
        maximum,
        minimum,
        peak_to_peak,
        skewness,
        kurt,
        energy
    ])

# ==========================================================
# SAVE FEATURE MATRIX
# ==========================================================

columns = [
    "Mean",
    "Std",
    "Variance",
    "RMS",
    "Maximum",
    "Minimum",
    "Peak_to_Peak",
    "Skewness",
    "Kurtosis",
    "Energy"
]

features = pd.DataFrame(features, columns=columns)

features["Class"] = labels["Class"]

features.to_csv(
    os.path.join(
        OUTPUT_DATA,
        "Time_Domain_Features.csv"
    ),
    index=False
)

print(features.head())

# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================

summary = features.groupby("Class").agg(["mean", "std"])

summary.to_csv(
    os.path.join(
        OUTPUT_TABLES,
        "Table_6_Time_Features_Statistics.csv"
    )
)

print(summary)

print("\nTime-domain features successfully extracted.")