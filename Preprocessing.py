import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# 1. Memuat dataset
file_path = "supervised-learning-eddowardo/dataset_encoded.csv"  # Sesuaikan dengan path yang benar

if not os.path.exists(file_path):
    print("File tidak ditemukan! Daftar file yang tersedia di direktori kerja:")
    print(os.listdir("supervised-learning-eddowardo"))
    raise FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(file_path)}")

df = pd.read_csv(file_path)

# 2. Encoding untuk fitur nonnumerik
# Memilih kolom yang bertipe objek (kategorikal)
categorical_features = df.select_dtypes(include=['object']).columns

# Menggunakan OrdinalEncoder untuk encoding data kategorikal
if len(categorical_features) > 0:
    encoder = OrdinalEncoder()
    df[categorical_features] = encoder.fit_transform(df[categorical_features])

# 3. Memisahkan fitur independen (X) dan target (Y)
target_column = "SalePrice"  # Sesuaikan dengan nama kolom target di dataset

if target_column not in df.columns:
    raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset.")

X = df.drop(columns=[target_column])
Y = df[target_column]

# 4. Membagi dataset menjadi training data (80%) dan testing data (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Jumlah data training: {len(X_train)}, Jumlah data testing: {len(X_test)}")

# 5. Visualisasi Distribusi Target (Y)
plt.figure(figsize=(10, 5))
sns.histplot(Y, bins=30, kde=True, color="blue")
plt.title("Distribusi Harga Rumah (SalePrice)")
plt.xlabel("Harga Rumah")
plt.ylabel("Frekuensi")
plt.show()

# 6. Visualisasi Korelasi antara Beberapa Fitur dengan Target
top_features = X.corrwith(Y).abs().sort_values(ascending=False).head(5).index

plt.figure(figsize=(12, 6))
for i, feature in enumerate(top_features):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=df[feature], y=Y, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("SalePrice")

plt.tight_layout()
plt.suptitle("Hubungan 5 Fitur Teratas dengan Harga Rumah", fontsize=14)
plt.show()
