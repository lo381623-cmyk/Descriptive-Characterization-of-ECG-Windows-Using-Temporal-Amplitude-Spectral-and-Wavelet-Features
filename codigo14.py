import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/features/Final_Feature_Matrix.csv")

features = ["Energy", "Spectral_Energy", "E_A4", "E_D3"]

for f in features:

    plt.figure(figsize=(6,4))

    df.boxplot(column=f, by="Class")

    plt.title(f"Boxplot of {f} by Class")
    plt.suptitle("")

    plt.xlabel("Class")
    plt.ylabel(f)

    plt.tight_layout()

    plt.savefig(
        f"results/figures/Boxplot_{f}.pdf",
        dpi=300
    )

    plt.close()

print("Boxplots done.")