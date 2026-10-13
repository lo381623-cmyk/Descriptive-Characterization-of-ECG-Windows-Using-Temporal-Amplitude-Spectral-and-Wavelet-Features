import os
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

TIME_PATH = "results/features/Time_Domain_Features.csv"
FFT_PATH = "results/features/FFT_Features.csv"
WAVELET_PATH = "results/features/Wavelet_Features.csv"
LABELS_PATH = "results/dataset/Balanced_Labels.csv"

OUTPUT_DIR = "results/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD FEATURES
# ==========================================================

time_df = pd.read_csv(TIME_PATH)
fft_df = pd.read_csv(FFT_PATH)
wavelet_df = pd.read_csv(WAVELET_PATH)
labels = pd.read_csv(LABELS_PATH)

print("Time:", time_df.shape)
print("FFT:", fft_df.shape)
print("Wavelet:", wavelet_df.shape)

# ==========================================================
# CONCATENATE FEATURES
# ==========================================================

feature_matrix = pd.concat(
    [time_df, fft_df, wavelet_df],
    axis=1
)

# Add class labels
feature_matrix["Class"] = labels["Class"]

# ==========================================================
# SAVE FINAL MATRIX
# ==========================================================

feature_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "Final_Feature_Matrix.csv"),
    index=False
)

print("\nFinal feature matrix shape:", feature_matrix.shape)
print("Feature matrix successfully created.")