import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

beats = pd.read_csv("results/dataset/Preprocessed_ECG_Beats.csv").values

beat = beats[0]

coeffs = pywt.wavedec(beat, "db4", level=4)

labels = ["A4", "D4", "D3", "D2", "D1"]

plt.figure(figsize=(10,8))

plt.subplot(6,1,1)
plt.plot(beat)
plt.title("Original ECG Beat")

for i, c in enumerate(coeffs):

    plt.subplot(6,1,i+2)
    plt.plot(c)
    plt.title(labels[i] if i < 5 else "Coefficients")

plt.tight_layout()

plt.savefig(
    "results/figures/Figure_9_Wavelet_Decomposition.pdf",
    dpi=300
)

plt.close()