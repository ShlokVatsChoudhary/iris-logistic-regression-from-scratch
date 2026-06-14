from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from preprocessing import z_scale_mean, z_scale_std, Z_scale_normalization

iris = load_iris()

X = iris.data
y_ini = iris.target

# Keep only Versicolor and Virginica
mask = y_ini != 0

X_binary = X[mask]
y_filtered = y_ini[mask]

# Versicolor: 1 -> 0
# Virginica: 2 -> 1
y_binary = (y_filtered == 2).astype(int)

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X_binary,
    y_binary,
    test_size=0.30,
    random_state=42,
    stratify=y_binary
)

# 15% validation, 15% test
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

X_train_scaled = Z_scale_normalization(X_train, z_scale_mean(X_train),z_scale_std(X_train))
X_cv_scaled = Z_scale_normalization(X_cv, z_scale_mean(X_cv),z_scale_std(X_cv))
X_test_scaled = Z_scale_normalization(X_test, z_scale_mean(X_test),z_scale_std(X_test))