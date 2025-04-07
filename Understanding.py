import pandas as pd

# Load dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Menghitung statistik deskriptif
stats = df.describe().transpose()

# Menambahkan median secara manual karena tidak termasuk dalam describe()
stats['median'] = df.median(numeric_only=True)

# Menampilkan jumlah nilai yang tersedia per kolom
stats['count'] = df.count()

# Menampilkan hasil
stats = stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
print(stats)
