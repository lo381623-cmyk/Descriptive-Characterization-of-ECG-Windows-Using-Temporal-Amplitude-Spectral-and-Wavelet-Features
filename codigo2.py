import os
import wfdb
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================

DATABASE_PATH = r"C:\Users\orduz\Desktop\EMG INV\dates2"

OUTPUT_TABLES = "results/tables"
OUTPUT_FIGURES = "results/figures"

os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# ==========================================================
# AAMI CLASSIFICATION
# ==========================================================

AAMI_CLASSES = {
    'N': 'Normal',
    'L': 'Normal',
    'R': 'Normal',
    'e': 'Normal',
    'j': 'Normal',

    'A': 'SVEB',
    'a': 'SVEB',
    'J': 'SVEB',
    'S': 'SVEB',

    'V': 'VEB',
    'E': 'VEB',

    'F': 'Fusion',

    '/': 'Unknown',
    'f': 'Unknown',
    'Q': 'Unknown'
}

# ==========================================================
# GET RECORDS
# ==========================================================

records = sorted([
    f.replace(".hea", "")
    for f in os.listdir(DATABASE_PATH)
    if f.endswith(".hea")
])

print(f"\nTotal records: {len(records)}")

# ==========================================================
# COUNT MIT SYMBOLS
# ==========================================================

symbol_counter = {}

for record in records:

    annotation = wfdb.rdann(
        os.path.join(DATABASE_PATH, record),
        "atr"
    )

    for symbol in annotation.symbol:

        symbol_counter[symbol] = symbol_counter.get(symbol, 0) + 1

# ==========================================================
# SAVE MIT SYMBOL TABLE
# ==========================================================

mit_table = (
    pd.DataFrame(
        symbol_counter.items(),
        columns=["MIT Symbol", "Count"]
    )
    .sort_values("Count", ascending=False)
)

mit_table.to_csv(
    os.path.join(
        OUTPUT_TABLES,
        "Table_2_MIT_Symbols.csv"
    ),
    index=False
)

print("\nMIT Symbols")
print(mit_table)

# ==========================================================
# CONVERT TO AAMI
# ==========================================================

aami_counter = {}

for symbol, count in symbol_counter.items():

    if symbol in AAMI_CLASSES:

        cls = AAMI_CLASSES[symbol]

    else:

        cls = "Other"

    aami_counter[cls] = aami_counter.get(cls, 0) + count

# ==========================================================
# SAVE AAMI TABLE
# ==========================================================

aami_table = (
    pd.DataFrame(
        aami_counter.items(),
        columns=["AAMI Class", "Count"]
    )
    .sort_values("Count", ascending=False)
)

aami_table.to_csv(
    os.path.join(
        OUTPUT_TABLES,
        "Table_3_AAMI_Distribution.csv"
    ),
    index=False
)

print("\nAAMI Distribution")
print(aami_table)

# ==========================================================
# FIGURE 2
# ORIGINAL CLASS DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    aami_table["AAMI Class"],
    aami_table["Count"]
)

plt.ylabel("Number of Beats")

plt.xlabel("AAMI Class")

plt.title("Original Distribution of Heartbeat Classes")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FIGURES,
        "Figure_2_Original_AAMI_Distribution.pdf"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# REPORT
# ==========================================================

print("\n========================================")
print("Original dataset successfully analyzed.")
print("========================================")

print("\nFiles generated:")

print("✓ Table_2_MIT_Symbols.csv")

print("✓ Table_3_AAMI_Distribution.csv")

print("✓ Figure_2_Original_AAMI_Distribution.pdf")