import numpy as np

#importing all the functions from preprocessing and model
from model import gradient_descent, compute_gradient_logistic, compute_cost, sigmoid, predict_logistic 

#importing the testing, cross-validation and training data which we have already scaled in preprocessing
from preprocessing import X_train, y_train, X_cv, y_cv, X_test, y_test, X_train_scaled, X_cv_scaled, X_test_scaled

#checking the shape of the array before running
print(X_train_scaled.shape)
print(X_cv_scaled.shape)
print(X_test_scaled.shape)
print(y_train.shape)
print(y_cv.shape)
print(y_test.shape)

#initial global variables 
W_ini = np.zeros(X_train_scaled.shape[1])
b_ini = 0
alpha = 0.1
num_iters = 1000

#running gradient descent to find best w
W_final, b_final, j_history = gradient_descent(
    X_train_scaled,
    y_train,
    W_ini,
    b_ini,
    num_iters,
    alpha
)

""" when alpha is 0.001 
iteration:    0 : cost 0.6925626114644907
iteration:  100 : cost 0.6380424954607259
iteration:  200 : cost 0.5905712971150613
iteration:  300 : cost 0.5491203473309815
iteration:  400 : cost 0.5127872768074286
iteration:  500 : cost 0.48079755923051476
iteration:  600 : cost 0.45249578825834735
iteration:  700 : cost 0.427332123333377
iteration:  800 : cost 0.4048473737430978
iteration:  900 : cost 0.38465865777330316"""

"""when alpha is 0.01
iteration:    0 : cost 0.6873199058530154
iteration:  100 : cost 0.3644342603139031
iteration:  200 : cost 0.24966790814883652
iteration:  300 : cost 0.19184120101564103
iteration:  400 : cost 0.15667751335378333
iteration:  500 : cost 0.1328601897184151
iteration:  600 : cost 0.11557812060365286
iteration:  700 : cost 0.10242583101786086
iteration:  800 : cost 0.09205994888743274
iteration:  900 : cost 0.08366794841042488"""

#Initial cost: 0.6925626114644907
#Final cost: 0.3666198044712175
#=> cost function is effectively decreasing

print("Final W:", W_final)
print("Final b:", b_final)

#Final W: [-0.22191046  0.23477241 -0.31512801 -0.30104216]
#Final b: -0.14798613367661043

#training the model using X_train 
y_train_pred = predict_logistic(X_train_scaled, W_final, b_final)
y_cv_pred = predict_logistic(X_cv_scaled, W_final, b_final)
print(y_cv_pred[:10])
print(y_cv[:10])

temp = 0
for i in range(y_cv.shape[0]):
    temp += abs(y_cv[i] - y_cv_pred[i])

error_rate = temp / y_cv.shape[0]
accuracy = 1 - error_rate

print(f"CV Error Rate: {error_rate * 100:.2f}%")
print(f"CV Accuracy: {accuracy * 100:.2f}%")

#error comes out to 0.00% in y_cv and y_cv_pred

y_test_pred = predict_logistic(X_test_scaled, W_final, b_final)

for actual, predicted in zip(y_test, y_test_pred):
    print(f"actual={actual}, predicted={predicted}")

test_accuracy = np.mean(y_test_pred == y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


