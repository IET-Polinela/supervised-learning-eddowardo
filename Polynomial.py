import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Setup visual
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# Folder output
os.makedirs("output_plots", exist_ok=True)

# === 1. Load dan scaling dataset TANPA outlier ===
df = pd.read_csv("dataset_no_outlier.csv")
df = df.select_dtypes(include=["number"])

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 2. Fungsi evaluasi & visualisasi (dengan simpan PNG) ===
def evaluate_model(y_true, y_pred, residuals, title_prefix):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"==== {title_prefix} ====")
    print(f"MSE      : {mse:.2f}")
    print(f"R² Score : {r2:.4f}")
    print()

    filename_prefix = title_prefix.replace(" ", "_").replace("(", "").replace(")", "").lower()

    # Scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    plt.title(f"{title_prefix} - Prediksi vs Aktual")
    plt.xlabel("Harga Aktual")
    plt.ylabel("Harga Prediksi")
    plt.tight_layout()
    plt.savefig(f"output_plots/{filename_prefix}_prediksi_vs_aktual.png")
    plt.close()

    # Residual plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{title_prefix} - Residual Plot")
    plt.xlabel("Harga Prediksi")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(f"output_plots/{filename_prefix}_residual_plot.png")
    plt.close()

    # Distribusi Residual
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=40, kde=True, color='blue')
    plt.title(f"{title_prefix} - Distribusi Residual")
    plt.xlabel("Residual")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(f"output_plots/{filename_prefix}_residual_distribution.png")
    plt.close()

    return mse, r2

# === 3. Linear Regression (Baseline)
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
residuals_linear = y_test - y_pred_linear
mse_linear, r2_linear = evaluate_model(y_test, y_pred_linear, residuals_linear, "Linear Regression")

# === 4. Polynomial Regression Degree = 2
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

model_poly2 = LinearRegression()
model_poly2.fit(X_train_poly2, y_train)
y_pred_poly2 = model_poly2.predict(X_test_poly2)
residuals_poly2 = y_test - y_pred_poly2
mse_poly2, r2_poly2 = evaluate_model(y_test, y_pred_poly2, residuals_poly2, "Polynomial Regression (Degree 2)")

# === 5. Polynomial Regression Degree = 3
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

model_poly3 = LinearRegression()
model_poly3.fit(X_train_poly3, y_train)
y_pred_poly3 = model_poly3.predict(X_test_poly3)
residuals_poly3 = y_test - y_pred_poly3
mse_poly3, r2_poly3 = evaluate_model(y_test, y_pred_poly3, residuals_poly3, "Polynomial Regression (Degree 3)")

# === 6. Ringkasan Perbandingan
summary = pd.DataFrame([
    {"Model": "Linear Regression", "MSE": mse_linear, "R2 Score": r2_linear},
    {"Model": "Polynomial Degree 2", "MSE": mse_poly2, "R2 Score": r2_poly2},
    {"Model": "Polynomial Degree 3", "MSE": mse_poly3, "R2 Score": r2_poly3}
])

print("=== Ringkasan Hasil Evaluasi ===")
print(summary.to_string(index=False))

# Simpan ringkasan ke CSV
summary.to_csv("output_plots/ringkasan_evaluasi.csv", index=False)
