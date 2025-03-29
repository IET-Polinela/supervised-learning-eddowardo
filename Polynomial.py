#soal 6 regression dengan outlier 
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset dengan outlier
file_path = "supervised-learning-eddowardo/dataset_encoded_final.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' tidak ditemukan! Pastikan dataset tersedia.")

df = pd.read_csv(file_path)

# 2. Menangani NaN dengan mengisi nilai median
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)  # Mengisi NaN dengan median

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
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 6. Linear Regression sebagai Baseline
model_linear = LinearRegression()
model_linear.fit(X_train, Y_train)
Y_pred_linear = model_linear.predict(X_test)

mse_linear = mean_squared_error(Y_test, Y_pred_linear)
r2_linear = r2_score(Y_test, Y_pred_linear)

# 7. Polynomial Regression (Degree 2 & 3)
degrees = [2, 3]
mse_values = [mse_linear]
r2_values = [r2_linear]
models = ['Linear Regression']

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, Y_train)
    Y_pred_poly = model_poly.predict(X_test_poly)

    mse_values.append(mean_squared_error(Y_test, Y_pred_poly))
    r2_values.append(r2_score(Y_test, Y_pred_poly))
    models.append(f'Poly d={degree}')

# 8. Visualisasi Perbandingan R2 Score
plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=r2_values, hue=models, palette='coolwarm', legend=False, errorbar=None)
plt.title("R2 Score - Linear & Polynomial Regression (Dengan Outlier)")
plt.xticks(rotation=45)
plt.ylabel("R2 Score")
plt.show()
