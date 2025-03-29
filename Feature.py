#soal 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Load dataset tanpa outlier
file_path_cleaned = "supervised-learning-eddowardo/dataset_no_outlier.csv"

# Cek apakah file ada
try:
    df = pd.read_csv(file_path_cleaned)
except FileNotFoundError:
    raise FileNotFoundError(f"File '{file_path_cleaned}' tidak ditemukan! Pastikan dataset tersedia.")

# 2. Pilih hanya fitur numerik untuk scaling
numerical_features = df.select_dtypes(include=[np.number]).columns
df_num = df[numerical_features]

# 3. Scaling menggunakan StandardScaler
scaler_standard = StandardScaler()
df_standard_scaled = scaler_standard.fit_transform(df_num)
df_standard_scaled = pd.DataFrame(df_standard_scaled, columns=numerical_features)

# 4. Scaling menggunakan MinMaxScaler
scaler_minmax = MinMaxScaler()
df_minmax_scaled = scaler_minmax.fit_transform(df_num)
df_minmax_scaled = pd.DataFrame(df_minmax_scaled, columns=numerical_features)

# 5. Visualisasi Histogram Sebelum dan Sesudah Scaling
plt.figure(figsize=(15, 10))

for i, feature in enumerate(numerical_features[:5]):  # Hanya menampilkan 5 fitur pertama agar lebih jelas
    plt.subplot(3, 5, i + 1)
    sns.histplot(df_num[feature], bins=30, kde=True, color="blue", label="Original", alpha=0.5)
    plt.title(f"{feature} (Original)")

    plt.subplot(3, 5, i + 6)
    sns.histplot(df_standard_scaled[feature], bins=30, kde=True, color="red", label="StandardScaled", alpha=0.5)
    plt.title(f"{feature} (StandardScaler)")

    plt.subplot(3, 5, i + 11)
    sns.histplot(df_minmax_scaled[feature], bins=30, kde=True, color="green", label="MinMaxScaled", alpha=0.5)
    plt.title(f"{feature} (MinMaxScaler)")

plt.tight_layout()
plt.show()
