import pandas as pd

df = pd.read_csv("results/features/Time_Domain_Features.csv")

print(df.groupby("Class")["Peak_to_Peak"].agg(["mean", "std"]))