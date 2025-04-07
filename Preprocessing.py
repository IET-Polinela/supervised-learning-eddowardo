import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# 1. Baca dataset
df = pd.read_csv('train.csv')

# 2. Pisahkan fitur nonnumerik (kategorikal)
cat_cols = df.select_dtypes(include='object').columns

# 3. Terapkan encoding untuk fitur-fitur nonnumerik
encoder = OrdinalEncoder()
df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))  # ubah ke string agar konsisten

# 4. Tampilkan hasil dataset setelah encoding
print("📄 Data setelah encoding:")
print(df.head())  # tampilkan 5 baris pertama

# 5. Pisahkan fitur (X) dan target (y)
# Ganti 'SalePrice' dengan nama kolom target jika berbeda
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# 6. Bagi dataset menjadi data train dan test (80% : 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Simpan hasil preprocessing ke file
df_encoded = pd.concat([X, y], axis=1)
df_encoded.to_csv('dataset_encoded.csv', index=False)

print("\n Data preprocessing selesai.")
print(f" Jumlah data training: {len(X_train)}")
print(f" Jumlah data testing : {len(X_test)}")
