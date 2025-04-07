import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Setup visual
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# Buat folder output jika belum ada
os.makedirs("output", exist_ok=True)

# === 1. Dataset DENGAN Outlier ===
df_outlier = pd.read_csv("train.csv")
df_outlier = df_outlier.select_dtypes(include=["number"]).dropna()

X_out = df_outlier.drop(columns=["SalePrice"])
y_out = df_outlier["SalePrice"]

X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(
    X_out, y_out, test_size=0.2, random_state=42
)

model_out = LinearRegression()
model_out.fit(X_train_out, y_train_out)

y_pred_out = model_out.predict(X_test_out)
residuals_out = y_test_out - y_pred_out

# === 2. Dataset TANPA Outlier ===
df_clean = pd.read_csv("dataset_no_outlier.csv")
df_clean = df_clean.select_dtypes(include=["number"])
X_clean = df_clean.drop(columns=["SalePrice"])
y_clean = df_clean["SalePrice"]

# --- 2A. Standard Scaler ---
scaler_standard = StandardScaler()
X_clean_std = pd.DataFrame(scaler_standard.fit_transform(X_clean), columns=X_clean.columns)

X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(
    X_clean_std, y_clean, test_size=0.2, random_state=42
)

model_std = LinearRegression()
model_std.fit(X_train_std, y_train_std)

y_pred_std = model_std.predict(X_test_std)
residuals_std = y_test_std - y_pred_std

# --- 2B. MinMax Scaler ---
scaler_minmax = MinMaxScaler()
X_clean_mm = pd.DataFrame(scaler_minmax.fit_transform(X_clean), columns=X_clean.columns)

X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(
    X_clean_mm, y_clean, test_size=0.2, random_state=42
)

model_mm = LinearRegression()
model_mm.fit(X_train_mm, y_train_mm)

y_pred_mm = model_mm.predict(X_test_mm)
residuals_mm = y_test_mm - y_pred_mm

# === 3. Evaluasi & Simpan Output + Tampilkan ===
def evaluate_model(y_true, y_pred, residuals, title_prefix, filename_prefix):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Simpan hasil evaluasi ke file
    with open(f"output/{filename_prefix}_metrics.txt", "w") as f:
        f.write(f"==== {title_prefix} ====\n")
        f.write(f"MSE      : {mse:.2f}\n")
        f.write(f"R² Score : {r2:.4f}\n")

    print(f"==== {title_prefix} ====")
    print(f"MSE      : {mse:.2f}")
    print(f"R² Score : {r2:.4f}\n")

    # Prediksi vs Aktual
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    plt.title(f"{title_prefix} - Prediksi vs Aktual")
    plt.xlabel("Harga Aktual")
    plt.ylabel("Harga Prediksi")
    plt.tight_layout()
    plt.savefig(f"output/{filename_prefix}_prediksi_vs_aktual.png")
    plt.show()  # <<< Tambahkan ini agar tampil

    # Residual Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{title_prefix} - Residual Plot")
    plt.xlabel("Harga Prediksi")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(f"output/{filename_prefix}_residual_plot.png")
    plt.show()  # <<< Tambahkan ini agar tampil

    # Distribusi Residual
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=40, kde=True, color='blue')
    plt.title(f"{title_prefix} - Distribusi Residual")
    plt.xlabel("Residual")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(f"output/{filename_prefix}_residual_distribution.png")
    plt.show()  # <<< Tambahkan ini agar tampil

# === Evaluasi semua model ===
evaluate_model(y_test_out, y_pred_out, residuals_out, "DENGAN Outlier", "with_outlier")
evaluate_model(y_test_std, y_pred_std, residuals_std, "TANPA Outlier + StandardScaler", "no_outlier_standard")
evaluate_model(y_test_mm, y_pred_mm, residuals_mm, "TANPA Outlier + MinMaxScaler", "no_outlier_minmax")
