import matplotlib.pyplot as plt
import seaborn as sns

# Data hasil prediksi (misalkan sudah dihitung sebelumnya)
Y_test_actual = [200000, 250000, 180000, 220000, 300000, 260000, 275000]  # Nilai aktual
Y_pred_linear = [195000, 245000, 185000, 215000, 290000, 255000, 270000]  # Linear Regression
Y_pred_poly2 = [198000, 248000, 182000, 218000, 295000, 258000, 273000]  # Polynomial d=2
Y_pred_poly3 = [200500, 250500, 180500, 220500, 298000, 260500, 275500]  # Polynomial d=3
Y_pred_knn3 = [190000, 240000, 175000, 210000, 285000, 250000, 265000]  # KNN k=3
Y_pred_knn5 = [193000, 243000, 177000, 213000, 288000, 252000, 268000]  # KNN k=5
Y_pred_knn7 = [195500, 245500, 179500, 215500, 290500, 255500, 270500]  # KNN k=7

# Fungsi untuk membuat scatter plot
def plot_actual_vs_predicted(y_actual, y_pred, title, color):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_actual, y=y_pred, color=color, alpha=0.7)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'k--', linewidth=2)  # Garis ideal
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Scatter plot untuk setiap model
plot_actual_vs_predicted(Y_test_actual, Y_pred_linear, "Linear Regression: Actual vs Predicted", "blue")
plot_actual_vs_predicted(Y_test_actual, Y_pred_poly2, "Polynomial Regression (Degree=2): Actual vs Predicted", "green")
plot_actual_vs_predicted(Y_test_actual, Y_pred_poly3, "Polynomial Regression (Degree=3): Actual vs Predicted", "red")
plot_actual_vs_predicted(Y_test_actual, Y_pred_knn3, "KNN Regression (k=3): Actual vs Predicted", "purple")
plot_actual_vs_predicted(Y_test_actual, Y_pred_knn5, "KNN Regression (k=5): Actual vs Predicted", "orange")
plot_actual_vs_predicted(Y_test_actual, Y_pred_knn7, "KNN Regression (k=7): Actual vs Predicted", "brown")
