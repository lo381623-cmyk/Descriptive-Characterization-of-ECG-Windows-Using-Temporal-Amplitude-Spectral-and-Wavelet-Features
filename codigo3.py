import os
import random
import wfdb
import numpy as np
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

DATABASE_PATH = r"C:\Users\orduz\Desktop\EMG INV\dates2"

OUTPUT_DATASET = "results/dataset"
OUTPUT_TABLES = "results/tables"

os.makedirs(OUTPUT_DATASET, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

# Samples before and after the R peak
PRE_SAMPLES = 250
POST_SAMPLES = 250

WINDOW = PRE_SAMPLES + POST_SAMPLES

# Number of beats per class
BEATS_PER_CLASS = 2000

# Reproducibility
random.seed(42)

# ==========================================================
# AAMI CLASSIFICATION
# ==========================================================

AAMI = {

    'N':'Normal',
    'L':'Normal',
    'R':'Normal',
    'e':'Normal',
    'j':'Normal',

    'A':'SVEB',
    'a':'SVEB',
    'J':'SVEB',
    'S':'SVEB',

    'V':'VEB',
    'E':'VEB'

}

TARGET_CLASSES = ["Normal","SVEB","VEB"]

# ==========================================================
# READ RECORDS
# ==========================================================

records = sorted([
    f.replace(".hea","")
    for f in os.listdir(DATABASE_PATH)
    if f.endswith(".hea")
])

print(f"\nRecords found: {len(records)}")

# ==========================================================
# STORE BEATS
# ==========================================================

beats = {
    "Normal": [],
    "SVEB": [],
    "VEB": []
}

metadata = {
    "Normal": [],
    "SVEB": [],
    "VEB": []
}

discarded = 0

# ==========================================================
# EXTRACT HEARTBEATS
# ==========================================================

for record in records:

    print(f"Processing record {record}")

    signal = wfdb.rdrecord(
        os.path.join(DATABASE_PATH, record)
    )

    annotation = wfdb.rdann(
        os.path.join(DATABASE_PATH, record),
        "atr"
    )

    ecg = signal.p_signal[:,0]

    for sample, symbol in zip(annotation.sample,
                              annotation.symbol):

        if symbol not in AAMI:
            continue

        label = AAMI[symbol]

        start = sample - PRE_SAMPLES
        end = sample + POST_SAMPLES

        if start < 0 or end >= len(ecg):
            discarded += 1
            continue

        beat = ecg[start:end]

        if len(beat) != WINDOW:
            discarded += 1
            continue

        beats[label].append(beat)

        metadata[label].append([
            record,
            sample,
            symbol,
            label
        ])

# ==========================================================
# DATASET STATISTICS
# ==========================================================

print("\nOriginal beats")

for cls in TARGET_CLASSES:

    print(cls, len(beats[cls]))

# ==========================================================
# BALANCE DATASET
# ==========================================================

balanced_beats = []
balanced_labels = []
balanced_metadata = []

statistics = []

for cls in TARGET_CLASSES:

    total = len(beats[cls])

    indices = random.sample(
        range(total),
        BEATS_PER_CLASS
    )

    for idx in indices:

        balanced_beats.append(
            beats[cls][idx]
        )

        balanced_labels.append(cls)

        balanced_metadata.append(
            metadata[cls][idx]
        )

    statistics.append([
        cls,
        total,
        BEATS_PER_CLASS
    ])

# ==========================================================
# CONVERT TO DATAFRAME
# ==========================================================

beats_df = pd.DataFrame(
    balanced_beats
)

labels_df = pd.DataFrame({
    "Class": balanced_labels
})

metadata_df = pd.DataFrame(
    balanced_metadata,
    columns=[
        "Record",
        "R_sample",
        "MIT_symbol",
        "AAMI_class"
    ]
)

statistics_df = pd.DataFrame(
    statistics,
    columns=[
        "Class",
        "Original beats",
        "Selected beats"
    ]
)

# ==========================================================
# SAVE FILES
# ==========================================================

beats_df.to_csv(
    os.path.join(
        OUTPUT_DATASET,
        "Balanced_ECG_Beats.csv"
    ),
    index=False
)

labels_df.to_csv(
    os.path.join(
        OUTPUT_DATASET,
        "Balanced_Labels.csv"
    ),
    index=False
)

metadata_df.to_csv(
    os.path.join(
        OUTPUT_DATASET,
        "Balanced_Metadata.csv"
    ),
    index=False
)

statistics_df.to_csv(
    os.path.join(
        OUTPUT_TABLES,
        "Table_4_Balanced_Dataset.csv"
    ),
    index=False
)

# ==========================================================
# REPORT
# ==========================================================

print("\n==========================================")

print("Balanced dataset successfully created.")

print("==========================================")

print(f"Discarded beats : {discarded}")

print(f"Total beats     : {len(balanced_beats)}")

print(f"Window length   : {WINDOW}")

print(f"Classes         : {TARGET_CLASSES}")

print("\nFiles generated:\n")

print("Balanced_ECG_Beats.csv")

print("Balanced_Labels.csv")

print("Balanced_Metadata.csv")

print("Table_4_Balanced_Dataset.csv")