import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_DATASET = "results/dataset/Balanced_ECG_Beats.csv"

OUTPUT_DATASET = "results/dataset"
OUTPUT_TABLES = "results/tables"
OUTPUT_FIGURES = "results/figures"

os.makedirs(OUTPUT_DATASET, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

beats = pd.read_csv(INPUT_DATASET)

print(f"Dataset shape: {beats.shape}")

beats = beats.values

# ==========================================================
# PREPROCESSING
# ==========================================================

processed = []

original_mean = []
original_std = []

processed_mean = []
processed_std = []

discarded = 0

for beat in beats:

    beat = np.asarray(beat, dtype=float)

    if np.isnan(beat).any():
        discarded += 1
        continue

    if np.isinf(beat).any():
        discarded += 1
        continue

    mean_before = np.mean(beat)
    std_before = np.std(beat)

    if std_before == 0:
        discarded += 1
        continue

    # Baseline correction
    beat = beat - mean_before

    # Z-score normalization
    beat = beat / std_before

    processed.append(beat)

    original_mean.append(mean_before)
    original_std.append(std_before)

    processed_mean.append(np.mean(beat))
    processed_std.append(np.std(beat))

processed = np.array(processed)

# ==========================================================
# SAVE DATASET
# ==========================================================

pd.DataFrame(processed).to_csv(
    os.path.join(
        OUTPUT_DATASET,
        "Preprocessed_ECG_Beats.csv"
    ),
    index=False
)

# ==========================================================
# STATISTICS
# ==========================================================

statistics = pd.DataFrame({

    "Property":[
        "Number of beats",
        "Discarded beats",
        "Average mean (original)",
        "Average std (original)",
        "Average mean (processed)",
        "Average std (processed)"
    ],

    "Value":[
        len(processed),
        discarded,
        np.mean(original_mean),
        np.mean(original_std),
        np.mean(processed_mean),
        np.mean(processed_std)
    ]

})

statistics.to_csv(

    os.path.join(
        OUTPUT_TABLES,
        "Table_5_Preprocessing_Statistics.csv"
    ),

    index=False

)

print(statistics)

# ==========================================================
# FIGURE 1
# ==========================================================

plt.figure(figsize=(12,5))

plt.plot(beats[0],label="Original")

plt.plot(processed[0],label="Preprocessed")

plt.xlabel("Samples")

plt.ylabel("Amplitude")

plt.title("Example of ECG Preprocessing")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_FIGURES,
        "Figure_5_Preprocessing_Comparison.pdf"
    ),

    dpi=300

)

plt.close()

# ==========================================================
# FIGURE 2
# ==========================================================

plt.figure(figsize=(10,5))

plt.hist(
    beats.flatten(),
    bins=80,
    alpha=0.5,
    label="Original"
)

plt.hist(
    processed.flatten(),
    bins=80,
    alpha=0.5,
    label="Preprocessed"
)

plt.xlabel("Amplitude")

plt.ylabel("Frequency")

plt.title("Amplitude Distribution Before and After Preprocessing")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        OUTPUT_FIGURES,
        "Figure_6_Amplitude_Distribution.pdf"
    ),

    dpi=300

)

plt.close()

print("\n===================================")
print("Preprocessing completed successfully")
print("===================================")