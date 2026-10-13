import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/features/FFT_Features.csv")
print(df["Spectral_Energy"].describe())
print(df["Spectral_Energy"].std() / df["Spectral_Energy"].mean())  # coef. de variación