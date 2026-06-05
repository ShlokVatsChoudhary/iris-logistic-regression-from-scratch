from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
#scikit learn used only for leading iris data, not for training model

iris = load_iris()
X = iris.data
y_ini = iris.target

# loaded prepackaged data of iris flower using scikit learn.
# X is [x1,x2,x3,x4] 
# x1 = sepal length
# x2 = sepal width
# x3 = petal length
# x4 = petal width

# Y is currently 0/1/2 for Setosa, Versicolor, and Virginica, but we need to convert to 0/1 i.e. not setosa vs setosa for our model here.

y_binary = (y_ini == 0).astype(int)

#print(X.shape) # (150, 4) => 150 examples of X

#splitting the data into test, cv and train


#Separate Train (70%) and Temporary/Remaining (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary) 

#Split the remaining 30% equally (50/50) into CV and Test
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# now we are calculating Mean, Standard daviation and scaled X for Z scale normalization

import numpy as np

#def z_scale_mean(X_train):
   # m = X_train.shape[0]  # number of rows (samples)
    #n = X_train.shape[1]  # number of columns (features = 4)
    #x_mean = np.zeros((1, n)) 
    
    #for x in range(n):  # Loop 4 times (0, 1, 2, 3)
     #   column_sum = X_train[:, x].sum()
      #  # Store the mean of this column in the correct slot
     #   x_mean[0, x] = column_sum / m  
    #return x_mean  meara likha hua

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

X_mean = z_scale_mean(X_train)
X_std = z_scale_std(X_train)

X_train_scaled = Z_scale_normalization(X_train,X_mean,X_std)
X_cv_scaled = Z_scale_normalization(X_cv,X_mean,X_std)
X_test_scaled = Z_scale_normalization(X_test,X_mean,X_std)

for x in range(X_cv_scaled.shape[0]):
    print(f"{X_cv_scaled[x]} : {y_cv[x]}")