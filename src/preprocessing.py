import numpy as np

def z_scale_mean(X_train):
    # axis=0 calculates the mean of each column
    # keepdims=True ensures the output shape stays (1, 4) instead of flattening to (4,)
    return np.mean(X_train, axis=0, keepdims=True)

def z_scale_std(X_train):
    # axis=0 calculates the standard deviation vertically down each column
    # keepdims=True keeps the shape at (1, 4) so it matches your x_mean shape perfectly
    return np.std(X_train, axis=0, keepdims=True)

def Z_scale_normalization(X, z_scale_mean, z_scale_std):
    return (X - z_scale_mean) / z_scale_std