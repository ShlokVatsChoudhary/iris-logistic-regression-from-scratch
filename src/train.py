import numpy as np

#importing all the functions from preprocessing and model
from model import gradient_descent, compute_gradient_logistic, compute_cost, sigmoid, predict_logistic, iterations_to_target

#importing the testing, cross-validation and training data which we have already scaled in preprocessing
from experiment_1_data import X_train, y_train, X_cv, y_cv, X_test, y_test, X_train_scaled, X_cv_scaled, X_test_scaled

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
alpha = 100
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

print(iterations_to_target(j_history, 0.002))

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


