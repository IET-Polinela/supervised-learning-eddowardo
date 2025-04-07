import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Baca dataset hasil encoding
df = pd.read_csv("dataset_encoded.csv")

# Hanya ambil fitur numerik (kecuali target)
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove("SalePrice")  # jika ingin mengecualikan target

# Fungsi untuk menghapus outlier menggunakan metode IQR
def remove_outliers_iqr(data, columns):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Dataset tanpa outlier
df_no_outlier = remove_outliers_iqr(df, numeric_cols)

# Ubah ke long format untuk boxplot gabungan
df_melted_before = df[numeric_cols].melt(var_name="Fitur", value_name="Nilai")
df_melted_after = df_no_outlier[numeric_cols].melt(var_name="Fitur", value_name="Nilai")

# Boxplot gabungan sebelum outlier dihapus
plt.figure(figsize=(20, 8))
sns.boxplot(data=df_melted_before, x="Fitur", y="Nilai", color="skyblue")
plt.title("Boxplot Semua Fitur Sebelum Menghapus Outlier")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("boxplot_sebelum_outlier.png", dpi=300)
plt.show()

# Boxplot gabungan setelah outlier dihapus
plt.figure(figsize=(20, 8))
sns.boxplot(data=df_melted_after, x="Fitur", y="Nilai", color="lightgreen")
plt.title("Boxplot Semua Fitur Setelah Menghapus Outlier (Metode IQR)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("boxplot_setelah_outlier.png", dpi=300)
plt.show()

# Simpan dataset
df.to_csv("dataset_with_outlier.csv", index=False)
df_no_outlier.to_csv("dataset_no_outlier.csv", index=False)

# Tampilkan jumlah data sebelum dan sesudah
print("Dataset dengan outlier: dataset_with_outlier.csv")
print("Dataset tanpa outlier: dataset_no_outlier.csv")
print("Jumlah data sebelum:", len(df))
print("Jumlah data setelah hapus outlier:", len(df_no_outlier))
