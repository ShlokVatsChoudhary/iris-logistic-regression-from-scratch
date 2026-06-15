from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from preprocessing import z_scale_mean, z_scale_std, Z_scale_normalization

#scikit learn used only for leading iris data, not for training model

iris = load_iris()
X = iris.data
y_ini = iris.target

# X is [x1,x2,x3,x4] 
# x1 = sepal length
# x2 = sepal width
# x3 = petal length
# x4 = petal width

# Y is currently 0/1/2 for Setosa, Versicolor, and Virginica, but we need to convert to 0/1 i.e. not setosa vs setosa for our model here.

y_binary = (y_ini == 0).astype(int)

#splitting the data into test, cv and train:

#Separate Train (70%) and Temporary/Remaining (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary) 

#Split the remaining 30% equally (50/50) into CV and Test
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# now we are calculating Mean, Standard daviation and scaled X for Z scale normalization

x_train_scaled = Z_scale_normalization(X_train, z_scale_mean(X_train), z_scale_std(X_train))
x_test_scaled = Z_scale_normalization(X_test, z_scale_mean(X_test), z_scale_std(X_test))
x_cv_scaled = Z_scale_normalization(X_cv, z_scale_mean(X_cv), z_scale_std(X_cv))

x = 0
y = 0
for i in range(y_binary.shape[0]):
    if (y_binary[i] == 0):
        x += 1
    else:
        y += 1
print("0 = ", x)
print("1 = ", y)

x = 0
y = 0
for i in range(y_train.shape[0]):
    if (y_train[i] == 0):
        x += 1
    else:
        y += 1
print("0 = ", x)
print("1 = ", y)

x = 0
y = 0
for i in range(y_cv.shape[0]):
    if (y_cv[i] == 0):
        x += 1
    else:
        y += 1
print("0 = ", x)
print("1 = ", y)

x = 0
y = 0
for i in range(y_test.shape[0]):
    if (y_test[i] == 0):
        x += 1
    else:
        y += 1
print("0 = ", x)
print("1 = ", y)