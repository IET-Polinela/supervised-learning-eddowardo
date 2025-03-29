import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load dataset tanpa outlier
file_path = "supervised-learning-eddowardo/dataset_no_outlier.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' tidak ditemukan! Pastikan dataset tersedia.")

df = pd.read_csv(file_path)

# 2. Menangani NaN dengan median
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# 3. Pisahkan fitur dan target
target_column = "SalePrice"
if target_column not in df.columns:
    raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset.")

X = df.drop(columns=[target_column])
Y = df[target_column]

# 4. Scaling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split dataset menjadi train dan test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 6. Model Linear Regression
model_linear = LinearRegression()
model_linear.fit(X_train, Y_train)
Y_pred_linear = model_linear.predict(X_test)

mse_values = [mean_squared_error(Y_test, Y_pred_linear)]
r2_values = [r2_score(Y_test, Y_pred_linear)]
models = ['Linear Regression']
predictions = {'Linear Regression': Y_pred_linear}

# 7. Polynomial Regression (Degree 2 & 3)
degrees = [2, 3]
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, Y_train)
    Y_pred_poly = model_poly.predict(X_test_poly)

    mse_values.append(mean_squared_error(Y_test, Y_pred_poly))
    r2_values.append(r2_score(Y_test, Y_pred_poly))
    models.append(f'Polynomial d={degree}')
    predictions[f'Polynomial d={degree}'] = Y_pred_poly

# 8. KNN Regression (K = 3, 5, 7)
k_values = [3, 5, 7]
for k in k_values:
    model_knn = KNeighborsRegressor(n_neighbors=k)
    model_knn.fit(X_train, Y_train)
    Y_pred_knn = model_knn.predict(X_test)

    mse_values.append(mean_squared_error(Y_test, Y_pred_knn))
    r2_values.append(r2_score(Y_test, Y_pred_knn))
    models.append(f'KNN k={k}')
    predictions[f'KNN k={k}'] = Y_pred_knn

# 9. Tabel Perbandingan MSE dan R² Score
df_comparison = pd.DataFrame({
    'Model': models,
    'MSE': mse_values,
    'R2 Score': r2_values
})

print("\n📊 Perbandingan Model:")
print(df_comparison)

# 10. Visualisasi Scatter Plot: Prediksi vs. Aktual
plt.figure(figsize=(10, 6))

for model, Y_pred in predictions.items():
    plt.scatter(Y_test, Y_pred, alpha=0.5, label=model)

plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='black', linestyle='--', linewidth=2)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Comparison: Actual vs Predicted Prices")
plt.legend()
plt.show()
