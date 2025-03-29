import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Data Understanding
# Path dataset yang benar
file_path = "supervised-learning-eddowardo/dataset_encoded.csv"

# Cek apakah file tersedia
if not os.path.exists(file_path):
    print("File tidak ditemukan! Daftar file yang tersedia di direktori kerja:")
    print(os.listdir("supervised-learning-eddowardo"))
    raise FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(file_path)}")

# Memuat dataset
df = pd.read_csv(file_path)

# Menampilkan statistik deskriptif lengkap
statistik_df = df.describe().T
statistik_df["median"] = df.median(numeric_only=True)
statistik_df["Q1"] = df.quantile(0.25, numeric_only=True)
statistik_df["Q2"] = df.quantile(0.50, numeric_only=True)  # Median sama dengan Q2
statistik_df["Q3"] = df.quantile(0.75, numeric_only=True)
statistik_df["missing_values"] = df.isnull().sum()
statistik_df["total_data"] = len(df)

# Menampilkan hasil
print(statistik_df)

# Analisis nilai yang hilang
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    print("\nFitur dengan nilai yang hilang:\n", missing_values)

    # Visualisasi Missing Values dengan Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Visualisasi Missing Values dalam Dataset")
    plt.show()
else:
    print("\nTidak ada nilai yang hilang dalam dataset.")

# Boxplot untuk melihat distribusi data
plt.figure(figsize=(12, 6))  # <<-- Bagian ini sudah diperbaiki!
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Distribusi Data dan Outlier di Setiap Fitur")
plt.tight_layout()  # Supaya tampilan tidak terpotong
plt.show()
