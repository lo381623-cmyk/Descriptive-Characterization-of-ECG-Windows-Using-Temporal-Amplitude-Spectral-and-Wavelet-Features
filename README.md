# Descriptive-Characterization-of-ECG-Windows-Using-Temporal-Amplitude-Spectral-and-Wavelet-Features

Code and supplementary material for the manuscript "Descriptive Characterization of ECG Windows Using Temporal, Amplitude, Spectral, and Wavelet Features" (anonymized for review).

The repository reproduces every table and figure of the manuscript from the public MIT-BIH Arrhythmia Database. The study is descriptive and exploratory: it does not train or evaluate a classifier, and no diagnostic or classification performance is claimed.

1. Overview of the pipeline
Count annotation symbols and group them into AAMI beat classes.
Extract 500-sample windows centered on annotated beats (N, SVEB, VEB) and draw 2,000 windows per class at random.
Preprocess each window (mean removal + z-score; no digital filtering).
Extract features: time-domain statistics, FFT magnitude-spectrum descriptors (including spectral entropy), and relative DWT sub-band energies.
Merge the features, remove constant features, and run PCA, t-SNE, and a correlation analysis (exploratory visualizations only).
2. Data

The raw data are not included in this repository. The MIT-BIH Arrhythmia Database (48 records, 47 subjects, 360 Hz) is publicly available on PhysioNet and can be downloaded with wfdb:

python
import wfdb
wfdb.dl_database("mitdb", "data/mitdb")   # downloads the 48 records

Then set DATABASE_PATH at the top of 01_symbol_counts.py and 02_build_dataset.py to the download folder (a relative path such as data/mitdb is recommended).

Included in this repository: data/Balanced_Metadata.csv, which identifies each of the 6,000 selected windows with the columns Record, R_sample (annotated sample index), MIT_symbol, and AAMI_class. With the raw records and this file, the exact set of windows can be reconstructed without re-running the random selection.

3. Requirements
Package	Version
Python	3.14.4
NumPy	2.4.4
SciPy	1.17.1
pandas	3.0.2
scikit-learn	1.8.0
PyWavelets (pywt)	1.8.0
wfdb	4.3.1
matplotlib	[fill in with pip freeze]
seaborn	[fill in with pip freeze]
bash
pip install -r requirements.txt

Results (in particular the t-SNE embedding) may differ slightly across library versions and platforms even with fixed seeds.

4. Repository structure
.
├── README.md
├── requirements.txt
├── src/
│   ├── 01_symbol_counts.py        # MIT symbol counts and AAMI distribution
│   ├── 02_build_dataset.py        # windows + class balancing (seed 42)
│   ├── 03_preprocess.py           # mean removal + z-score
│   ├── 04_time_features.py        # time-domain features
│   ├── 05_fft_features.py         # FFT magnitude-spectrum features
│   ├── 06_wavelet_features.py     # relative DWT sub-band energies
│   ├── 07_merge_features.py       # builds Final_Feature_Matrix.csv
│   ├── 08_pca_tsne_corr.py        # PCA, t-SNE, correlation matrix
│   └── figures/
│       ├── fig_ecg_segment.py             # manuscript Figure 1
│       ├── fig_representative_beats.py    # manuscript Figure 3
│       └── fig_wavelet_decomposition.py   # manuscript Figure 5
├── data/
│   └── Balanced_Metadata.csv
└── results/
    ├── dataset/    # intermediate CSV files
    ├── features/   # feature tables
    ├── tables/     # summary tables
    └── figures/    # PDF figures
5. How to run

Run the scripts in numerical order from the repository root:

bash
python src/01_symbol_counts.py
python src/02_build_dataset.py
python src/03_preprocess.py
python src/04_time_features.py
python src/05_fft_features.py
python src/06_wavelet_features.py
python src/07_merge_features.py
python src/08_pca_tsne_corr.py
Script	Main inputs	Main outputs
01_symbol_counts.py	raw records	Table_2_MIT_Symbols.csv, Table_3_AAMI_Distribution.csv, Figure_2_Original_AAMI_Distribution.pdf
02_build_dataset.py	raw records	Balanced_ECG_Beats.csv, Balanced_Labels.csv, Balanced_Metadata.csv, Table_4_Balanced_Dataset.csv
03_preprocess.py	Balanced_ECG_Beats.csv	Preprocessed_ECG_Beats.csv, Table_5_Preprocessing_Statistics.csv, Figure_5_Preprocessing_Comparison.pdf, Figure_6_Amplitude_Distribution.pdf
04_time_features.py	Preprocessed_ECG_Beats.csv, Balanced_Labels.csv	Time_Domain_Features.csv, Table_6_Time_Features_Statistics.csv
05_fft_features.py	Preprocessed_ECG_Beats.csv, Balanced_Labels.csv	FFT_Features.csv, Table_7_FFT_Features.csv, Figure_8_Average_FFT.pdf
06_wavelet_features.py	Preprocessed_ECG_Beats.csv, Balanced_Labels.csv	Wavelet_Features.csv, Table_8_Wavelet_Features.csv
07_merge_features.py	the three feature tables	Final_Feature_Matrix.csv
08_pca_tsne_corr.py	Final_Feature_Matrix.csv	Figure_10_PCA.pdf, Figure_11_tSNE.pdf, Figure_12_Correlation_Heatmap.pdf

The three feature tables are aligned row by row because all of them are computed from the same Preprocessed_ECG_Beats.csv in the same order.

6. Correspondence with the manuscript

The file names produced by the scripts do not follow the manuscript numbering. The correspondence is:

Manuscript item	Generated file
Figure 1 (10 s segment, record 100)	figures/fig_ecg_segment.py
Figure 2 (original vs. preprocessed window)	Figure_5_Preprocessing_Comparison.pdf
Figure 3 (example windows per class)	Figure_7_Representative_Beats.pdf (first selected window of each class)
Figure 4 (average FFT magnitude spectrum)	Figure_8_Average_FFT.pdf
Figure 5 (db4 decomposition, level 4)	Figure_9_Wavelet_Decomposition.pdf (first window, a Normal beat)
Figure 6 (PCA)	Figure_10_PCA.pdf
Figure 7 (t-SNE)	Figure_11_tSNE.pdf
Figure 8 (correlation matrix)	Figure_12_Correlation_Heatmap.pdf
Table 1 (time-domain features)	Table_6_Time_Features_Statistics.csv
Table 2 (relative wavelet energies)	Table_8_Wavelet_Features.csv
Spectral descriptors reported in Section 5.4	Table_7_FFT_Features.csv
Class counts reported in Section 5.2	Table_2_MIT_Symbols.csv, Table_3_AAMI_Distribution.csv, Table_4_Balanced_Dataset.csv
7. Processing configuration
Element	Configuration
Database	MIT-BIH Arrhythmia Database: 48 records from 47 subjects (records 201 and 202 are from the same subject), 360 Hz. All records were used; none was excluded, including the four records with paced rhythm (102, 104, 107, 217)
Lead	First signal channel of each record (MLII in 45 records, V5 in 3), physical units (mV)
Class grouping	AAMI beat classes, using the symbol grouping applied to MIT-BIH by de Chazal et al. (2004): N (N, L, R, e, j), SVEB (A, a, J, S), VEB (V, E), F (F), Q (/, f, Q). Only N, SVEB, and VEB were used. No edition of the AAMI EC57 standard is claimed
Annotation counts	90,631 N; 2,781 SVEB; 7,236 VEB; 803 F; 8,043 Q; 3,153 non-beat annotations
Window	500 samples (1.389 s) from r - 250 to r + 249, where r is the annotated sample. Beats whose window fell outside the record were discarded (72 N, 1 SVEB, 3 VEB). Available: 90,559 N; 2,780 SVEB; 7,233 VEB
Window overlap	Not treated. 1,316 of the 6,000 windows overlap with another selected window of the same record
Selection	2,000 beats per class, drawn at random without replacement, pooling all records (random.seed(42), random.sample)
Filtering	None
Normalization	Mean removal and z-score per window, with the population standard deviation (numpy.std, ddof=0). Quality control for NaN, infinite values, and zero variance (0 windows discarded)
Time-domain features	Maximum, Minimum, Peak-to-Peak (PP), skewness (population moments, scipy.stats.skew), and excess kurtosis (Fisher, scipy.stats.kurtosis), all computed on the normalized window
Spectral features	numpy.fft.rfft without tapering or zero-padding: 251 bins, 0.72 Hz resolution (0-180 Hz). Dominant frequency, spectral centroid and bandwidth (magnitude-weighted), and spectral entropy (p = abs(X) / sum(abs(X)), natural logarithm, in nats)
DWT	Daubechies-4, 4 levels (pywt.wavedec, default symmetric extension). Relative energies of A4, D4, D3, D2, D1, normalized by the sum of the five sub-band energies
Excluded from PCA/t-SNE/correlation	Mean, Std, Variance, RMS, Energy (constant under z-score) and Spectral Energy (numerically constant: mean 125,000.6, SD 1.19, by Parseval)
Feature vector (14)	Maximum, Minimum, Peak_to_Peak, Skewness, Kurtosis, Dominant_Frequency, Spectral_Centroid, Bandwidth, Spectral_Entropy, E_A4, E_D4, E_D3, E_D2, E_D1
Standardization	StandardScaler (zero mean, unit variance) before PCA and t-SNE
PCA	2 components (scikit-learn); explained variance 40.40 % and 18.67 % (59.08 % cumulative)
t-SNE	2 components, perplexity 30, init="pca", random_state=42, other parameters at their defaults
Correlation	Pearson correlation of the 14 features
8. Known limitations (see the manuscript, Discussion)
Windows from the same subject appear within the same class, so observations are not independent. Record 232 supplies 1,002 of the 2,000 SVEB windows.
500-sample windows may include parts of adjacent beats, and overlapping windows may belong to different classes.
The class-balanced design (2,000 per class) does not reflect natural class prevalence, and class N includes bundle-branch-block and escape beats.
The lead is not homogeneous (MLII in 45 records, V5 in 3).
No digital filtering was applied: baseline wander and power-line interference (visible near 60 Hz in the average spectrum) may contribute to the spectral descriptors and to the energies of the A4 and D2 sub-bands.
Normalization removes absolute amplitude.
No inferential statistics, confidence intervals, or generalization claims are made. Any predictive evaluation should split the data by patient.
