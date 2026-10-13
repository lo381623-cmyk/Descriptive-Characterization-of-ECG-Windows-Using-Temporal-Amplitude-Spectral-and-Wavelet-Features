import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# CONFIGURACIÓN
# ================================

database_path = r"C:\Users\orduz\Desktop\EMG INV\dates2"          # Cambiar si es necesario

output_tables = "results/tables"
output_figures = "results/figures"

os.makedirs(output_tables, exist_ok=True)
os.makedirs(output_figures, exist_ok=True)

# ================================
# LISTAR REGISTROS
# ================================

records = sorted(list(set(
    file.split(".")[0]
    for file in os.listdir(database_path)
    if file.endswith(".hea")
)))

print(f"Total records found: {len(records)}")

# ================================
# LEER INFORMACIÓN DE CADA REGISTRO
# ================================

summary = []

for record in records:

    record_path = os.path.join(database_path, record)

    signal = wfdb.rdrecord(record_path)

    fs = signal.fs

    n_samples = signal.sig_len

    duration = n_samples / fs / 60

    leads = signal.n_sig

    summary.append([
        record,
        n_samples,
        duration,
        fs,
        leads
    ])

summary_df = pd.DataFrame(
    summary,
    columns=[
        "Record",
        "Samples",
        "Duration (min)",
        "Sampling Frequency (Hz)",
        "Leads"
    ]
)

summary_df.to_csv(
    os.path.join(
        output_tables,
        "Table_1_Dataset_Characteristics.csv"
    ),
    index=False
)

print(summary_df.head())

# ================================
# RESUMEN GENERAL
# ================================

record0 = wfdb.rdrecord(os.path.join(database_path, records[0]))

general = pd.DataFrame({

    "Property":[
        "Database",
        "Records",
        "Sampling Frequency (Hz)",
        "Leads",
        "Average Duration (min)",
        "Average Samples"
    ],

    "Value":[
        "MIT-BIH Arrhythmia",
        len(records),
        record0.fs,
        record0.n_sig,
        round(summary_df["Duration (min)"].mean(),2),
        int(summary_df["Samples"].mean())
    ]

})

general.to_csv(
    os.path.join(
        output_tables,
        "Dataset_Summary.csv"
    ),
    index=False
)

print(general)

# ================================
# FIGURA 1
# ECG REPRESENTATIVO
# ================================

record100 = wfdb.rdrecord(os.path.join(database_path, "100"))

fs = record100.fs

signal = record100.p_signal[:,0]

seconds = 10

samples = fs * seconds

time = np.arange(samples) / fs

plt.figure(figsize=(12,4))

plt.plot(time, signal[:samples], linewidth=1)

plt.xlabel("Time (s)")

plt.ylabel("Amplitude (mV)")

plt.title("Representative ECG Segment (Record 100)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_figures,
        "Figure_1_Representative_ECG.pdf"
    ),
    dpi=300
)

plt.close()

# ================================
# FIGURA 2
# Duración de registros
# ================================

plt.figure(figsize=(12,5))

plt.bar(
    summary_df["Record"],
    summary_df["Duration (min)"]
)

plt.xticks(rotation=90)

plt.ylabel("Duration (minutes)")

plt.xlabel("Record")

plt.title("Duration of MIT-BIH Records")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_figures,
        "Figure_2_Record_Duration.pdf"
    ),
    dpi=300
)

plt.close()

# ================================
# GUARDAR REPORTE
# ================================

with open("results/dataset_summary.txt","w") as f:

    f.write("MIT-BIH Arrhythmia Database\n")
    f.write("============================\n\n")

    f.write(f"Number of records: {len(records)}\n")

    f.write(f"Sampling frequency: {record0.fs} Hz\n")

    f.write(f"Number of leads: {record0.n_sig}\n")

    f.write(f"Average duration: {summary_df['Duration (min)'].mean():.2f} min\n")

    f.write(f"Average samples: {summary_df['Samples'].mean():.0f}\n")

print("\nDataset description completed successfully.")