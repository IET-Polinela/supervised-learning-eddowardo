import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer

# ----------------------------
# 0. Setup folder output
# ----------------------------
output_dir = "output_plots"##
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# 1. Load Data
# ----------------------------
data = pd.read_csv('dataset_encoded.csv')

# ----------------------------
# 2. Pisahkan fitur dan target
# ----------------------------
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# ----------------------------
# 3. Imputasi NaN dengan mean
# ----------------------------
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# ----------------------------
# 5. Standardisasi
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 6. Linear Regression
# ----------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# ----------------------------
# 7. Polynomial Regression (degree = 2)
# ----------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# ----------------------------
# 8. KNN Regression (K = 3, 5, 7)
# ----------------------------
knn_results = {}
for k in [3, 5, 7]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)
    knn_results[k] = {'MSE': mse_knn, 'R2': r2_knn, 'y_pred': y_pred_knn}

# ----------------------------
# 9. Tampilkan Hasil Evaluasi sebagai Tabel
# ----------------------------
results = {
    'Model': ['Linear Regression', 'Polynomial Regression', 'KNN (K=3)', 'KNN (K=5)', 'KNN (K=7)'],
    'MSE': [mse_lin, mse_poly, knn_results[3]['MSE'], knn_results[5]['MSE'], knn_results[7]['MSE']],
    'R² Score': [r2_lin, r2_poly, knn_results[3]['R2'], knn_results[5]['R2'], knn_results[7]['R2']]
}
df_results = pd.DataFrame(results)

print("\n=== Evaluasi Model ===")
print(df_results.to_string(index=False))

# ----------------------------
# 10. Visualisasi Perbandingan MSE dan R² Score
# ----------------------------
sns.set(style="whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Perbandingan Kinerja Model (MSE dan R² Score)", fontsize=16, weight='bold')

# Plot MSE
sns.barplot(x='Model', y='MSE', hue='Model', data=df_results, ax=axs[0], palette='Blues_d', legend=False)
axs[0].set_title('Mean Squared Error (MSE)')
axs[0].tick_params(axis='x', rotation=45)
for i, val in enumerate(df_results['MSE']):
    axs[0].text(i, val + 5000, f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

# Plot R² Score
sns.barplot(x='Model', y='R² Score', hue='Model', data=df_results, ax=axs[1], palette='Greens_d', legend=False)
axs[1].set_title('R² Score')
axs[1].tick_params(axis='x', rotation=45)
for i, val in enumerate(df_results['R² Score']):
    axs[1].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, "barplot_mse_r2.png"))
plt.show()

# ----------------------------
# 11. Visualisasi Prediksi vs Aktual
# ----------------------------
plt.figure(figsize=(16, 10))
plt.suptitle('Prediksi vs Aktual (Semakin Dekat ke Garis Merah = Semakin Akurat)', fontsize=16, weight='bold')

def plot_pred_actual(index, y_true, y_pred, title, color):
    plt.subplot(2, 3, index)
    plt.scatter(y_true, y_pred, alpha=0.4, color=color)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(title)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.grid(True)

# Plot untuk setiap model
plot_pred_actual(1, y_test, y_pred_lin, 'Linear Regression', 'blue')
plot_pred_actual(2, y_test, y_pred_poly, 'Polynomial Regression', 'green')
plot_pred_actual(4, y_test, knn_results[3]['y_pred'], 'KNN (K=3)', 'orange')
plot_pred_actual(5, y_test, knn_results[5]['y_pred'], 'KNN (K=5)', 'purple')
plot_pred_actual(6, y_test, knn_results[7]['y_pred'], 'KNN (K=7)', 'brown')

# Panel tengah kosong berisi info
plt.subplot(2, 3, 3)
plt.axis('off')
plt.text(0.5, 0.5, 'Garis Merah = Prediksi Ideal\ny = x', ha='center', va='center', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(output_dir, "scatter_prediksi_vs_aktual.png"))
plt.show()
