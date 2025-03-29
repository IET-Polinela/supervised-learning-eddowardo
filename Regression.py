#soal ke 5 Linear Regression tanpa Outlier & dengan Scaling
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Memuat dataset yang telah diproses (tanpa outlier dan sudah scaling)
file_path = "supervised-learning-eddowardo/dataset_encoded_final.csv"

if not os.path.exists(file_path):
    print("File tidak ditemukan! Daftar file yang tersedia di direktori kerja:")
    print(os.listdir("supervised-learning-eddowardo"))
    raise FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(file_path)}")

df_clean = pd.read_csv(file_path)

# 2. Konversi fitur kategorikal ke numerik jika masih ada
categorical_features = df_clean.select_dtypes(include=['object']).columns

if len(categorical_features) > 0:
    encoder = LabelEncoder()
    for col in categorical_features:
        df_clean[col] = encoder.fit_transform(df_clean[col])

# 3. Menangani missing values (mengisi dengan median)
imputer = SimpleImputer(strategy="median")
df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)

# 4. Pastikan tidak ada NaN setelah imputasi
if df_clean.isnull().sum().sum() > 0:
    print("Masih ada NaN setelah imputasi. Menghapus baris yang mengandung NaN...")
    df_clean.dropna(inplace=True)

# 5. Pisahkan fitur (X) dan target (Y)
target_column = "SalePrice"
X_clean = df_clean.drop(columns=[target_column])
Y_clean = df_clean[target_column]

# 6. Scaling menggunakan StandardScaler dan MinMaxScaler
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X_clean)
X_standard_df = pd.DataFrame(X_standard, columns=X_clean.columns)

scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X_clean)
X_minmax_df = pd.DataFrame(X_minmax, columns=X_clean.columns)

# 7. Split data untuk training & testing (80% training, 20% testing)
X_train_clean, X_test_clean, Y_train_clean, Y_test_clean = train_test_split(X_standard_df, Y_clean, test_size=0.2, random_state=42)

# 8. Pastikan tidak ada NaN sebelum training
assert not np.any(pd.isnull(X_train_clean)), "X_train_clean masih mengandung NaN!"
assert not np.any(pd.isnull(X_test_clean)), "X_test_clean masih mengandung NaN!"

# 9. Training model Linear Regression
model_clean = LinearRegression()
model_clean.fit(X_train_clean, Y_train_clean)

# 10. Prediksi
Y_pred_clean = model_clean.predict(X_test_clean)

# 11. Evaluasi model
mse_clean = mean_squared_error(Y_test_clean, Y_pred_clean)
r2_clean = r2_score(Y_test_clean, Y_pred_clean)

print(f"Mean Squared Error (MSE) setelah scaling: {mse_clean:.2f}")
print(f"R² Score setelah scaling: {r2_clean:.4f}")

# 12. Visualisasi hasil prediksi
plt.figure(figsize=(12, 5))

# Scatter plot: Nilai aktual vs Prediksi
plt.subplot(1, 2, 1)
sns.scatterplot(x=Y_test_clean, y=Y_pred_clean, alpha=0.5)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Prediksi vs Nilai Aktual (Setelah Scaling)")

# Residual plot
plt.subplot(1, 2, 2)
sns.histplot(Y_test_clean - Y_pred_clean, bins=30, kde=True, color="red")
plt.xlabel("Residuals (Error)")
plt.title("Distribusi Residual (Setelah Scaling)")

plt.tight_layout()
plt.show()
