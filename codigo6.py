import matplotlib.pyplot as plt
import pandas as pd

beats = pd.read_csv(
    "results/dataset/Preprocessed_ECG_Beats.csv"
).values

labels = pd.read_csv(
    "results/dataset/Balanced_Labels.csv"
)

plt.figure(figsize=(12,8))

classes = ["Normal","SVEB","VEB"]

for i, c in enumerate(classes):

    idx = labels[labels["Class"]==c].index[0]

    plt.subplot(3,1,i+1)

    plt.plot(beats[idx])

    plt.title(c)

    plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/Figure_7_Representative_Beats.pdf",
    dpi=300
)

plt.close()