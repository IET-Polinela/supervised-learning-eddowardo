import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Memuat dataset tanpa outlier
file_path = "supervised-learning-eddowardo/dataset_encoded_final.csv"

if not os.path.exists(file_path):
    print("File tidak ditemukan! Daftar file yang tersedia di direktori kerja:")
    print(os.listdir())  # Menampilkan isi direktori root
    if os.path.exists("supervised-learning-eddowardo"):
        print("Isi folder 'supervised-learning-eddowardo':", os.listdir("supervised-learning-eddowardo"))
    raise FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(file_path)}")

df = pd.read_csv(file_path)

# 2. Memisahkan fitur independen (X) dan target (Y)
target_column = "SalePrice"  # Sesuaikan dengan nama kolom target di dataset

if target_column not in df.columns:
    raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset.")

X = df.drop(columns=[target_column])
Y = df[target_column]

# 3. Menerapkan StandardScaler
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)
X_standard_df = pd.DataFrame(X_standard, columns=X.columns)

# 4. Menerapkan MinMaxScaler
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
X_minmax_df = pd.DataFrame(X_minmax, columns=X.columns)

# 5. Visualisasi Perbandingan Distribusi Sebelum dan Sesudah Scaling
num_features = min(3, X.shape[1])  # Pilih maksimal 3 fitur untuk visualisasi
fig, axes = plt.subplots(num_features, 3, figsize=(15, 5 * num_features))

for i in range(num_features):
    feature = X.columns[i]

    # Histogram sebelum scaling
    sns.histplot(X[feature], bins=30, kde=True, ax=axes[i, 0], color="blue")
    axes[i, 0].set_title(f"Distribusi {feature} Sebelum Scaling")
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].set_ylabel("Frekuensi")

    # Histogram setelah StandardScaler
    sns.histplot(X_standard_df[feature], bins=30, kde=True, ax=axes[i, 1], color="green")
    axes[i, 1].set_title(f"Distribusi {feature} Setelah StandardScaler")
    axes[i, 1].set_xlabel(feature)
    axes[i, 1].set_ylabel("Frekuensi")

    # Histogram setelah MinMaxScaler
    sns.histplot(X_minmax_df[feature], bins=30, kde=True, ax=axes[i, 2], color="red")
    axes[i, 2].set_title(f"Distribusi {feature} Setelah MinMaxScaler")
    axes[i, 2].set_xlabel(feature)
    axes[i, 2].set_ylabel("Frekuensi")

plt.tight_layout()
plt.show()
