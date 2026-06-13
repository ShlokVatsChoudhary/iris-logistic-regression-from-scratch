import numpy as np

"""def z_scale_mean(X_train):
     m = X_train.shape[0]  # number of rows (samples)
    n = X_train.shape[1]  # number of columns (features = 4)
    x_mean = np.zeros((1, n)) 
    
    for x in range(n):  # Loop 4 times (0, 1, 2, 3)
       column_sum = X_train[:, x].sum()
        # Store the mean of this column in the correct slot
       x_mean[0, x] = column_sum / m  
    #return x_mean  meara likha hua """

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