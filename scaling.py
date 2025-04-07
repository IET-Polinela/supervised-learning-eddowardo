import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

# Setup tampilan
sns.set(style="darkgrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14
})

# Baca dataset
df = pd.read_csv("dataset_no_outlier.csv")

# Ambil fitur numerik, kecuali target
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove("SalePrice")
df_numeric = df[numeric_cols]

# Scaling
scaler_std = StandardScaler()
df_std_scaled = pd.DataFrame(scaler_std.fit_transform(df_numeric), columns=numeric_cols)

scaler_mm = MinMaxScaler()
df_mm_scaled = pd.DataFrame(scaler_mm.fit_transform(df_numeric), columns=numeric_cols)

# Warna gradasi
colors = sns.color_palette("husl", len(numeric_cols[:10]))

# Fungsi visualisasi
def plot_all_features(data, title, filename):
    plt.figure(figsize=(14, 6))
    for idx, col in enumerate(data.columns[:10]):  # Ambil hanya 10 fitur pertama
        sns.histplot(
            data[col],
            bins=40,
            color=colors[idx],
            kde=False,
            stat="frequency",
            element="step",
            alpha=0.5,
            label=col
        )

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, title="Features")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Visualisasi & Simpan gambar
plot_all_features(df_numeric, "Original Data Distribution", "histogram_original.png")
plot_all_features(df_std_scaled, "StandardScaler Data Distribution", "histogram_standard_scaled.png")
plot_all_features(df_mm_scaled, "MinMaxScaler Data Distribution", "histogram_minmax_scaled.png")
