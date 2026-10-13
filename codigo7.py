import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ==========================================================
# CONFIGURATION
# ==========================================================

FS = 360

INPUT_DATA = "results/dataset/Preprocessed_ECG_Beats.csv"
LABELS = "results/dataset/Balanced_Labels.csv"

OUTPUT_FEATURES = "results/features"
OUTPUT_TABLES = "results/tables"
OUTPUT_FIGURES = "results/figures"

os.makedirs(OUTPUT_FEATURES, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

beats = pd.read_csv(INPUT_DATA).values
labels = pd.read_csv(LABELS)

features = []

spectra = {
    "Normal": [],
    "SVEB": [],
    "VEB": []
}

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

for beat, label in zip(beats, labels["Class"]):

    fft = np.abs(np.fft.rfft(beat))

    freqs = np.fft.rfftfreq(len(beat), d=1/FS)

    dominant_frequency = freqs[np.argmax(fft)]

    centroid = np.sum(freqs * fft) / np.sum(fft)

    bandwidth = np.sqrt(
        np.sum(((freqs-centroid)**2)*fft)/np.sum(fft)
    )

    spectral_energy = np.sum(fft**2)

    p = fft / np.sum(fft)

    spectral_entropy = entropy(p)

    features.append([
        dominant_frequency,
        centroid,
        bandwidth,
        spectral_energy,
        spectral_entropy
    ])

    spectra[label].append(fft)

# ==========================================================
# SAVE FEATURES
# ==========================================================

columns = [
    "Dominant_Frequency",
    "Spectral_Centroid",
    "Bandwidth",
    "Spectral_Energy",
    "Spectral_Entropy"
]

df = pd.DataFrame(features, columns=columns)

df["Class"] = labels["Class"]

df.to_csv(
    os.path.join(
        OUTPUT_FEATURES,
        "FFT_Features.csv"
    ),
    index=False
)

# ==========================================================
# SUMMARY TABLE
# ==========================================================

summary = df.groupby("Class").agg(["mean","std"])

summary.to_csv(
    os.path.join(
        OUTPUT_TABLES,
        "Table_7_FFT_Features.csv"
    )
)

# ==========================================================
# FIGURE
# ==========================================================

plt.figure(figsize=(10,6))

for cls in ["Normal","SVEB","VEB"]:

    avg = np.mean(spectra[cls], axis=0)

    plt.plot(freqs, avg, label=cls)

plt.xlim(0,80)

plt.xlabel("Frequency (Hz)")

plt.ylabel("Magnitude")

plt.title("Average FFT Spectrum")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FIGURES,
        "Figure_8_Average_FFT.pdf"
    ),
    dpi=300
)

plt.close()

print("FFT features extracted successfully.")