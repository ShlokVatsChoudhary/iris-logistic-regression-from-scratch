import numpy as np
import matplotlib.pyplot as plt

#importing all the required functions
from model import gradient_descent, predict_logistic, compute_cost
from metrics import f1_score, recall_score, precision_score, manual_accuracy, confusion_matrix

#importing all the required variables 
from experiment_2_data import X_test_scaled, y_test, X_cv_scaled, y_cv, X_train_scaled, y_train

W_ini = np.zeros(X_train_scaled.shape[1]) #[0,0,0,0]
b_ini = 0 #scalar 
alpha = 0.01
num_iters = 5000

W_final, b_final, j_history = gradient_descent(
    X_train_scaled,
    y_train,
    W_ini,
    b_ini,
    num_iters,
    alpha
)
print(W_final)
print(b_final)

#training the model:

y_cv_pred = predict_logistic(X_cv_scaled,W_final,b_final)

train_cost = compute_cost(
    X_train_scaled,
    y_train,
    W_final,
    b_final
)

#testing the model
cv_cost = compute_cost(
    X_cv_scaled,
    y_cv,
    W_final,
    b_final
)

print("Final training cost:", train_cost)
print("Final CV cost:", cv_cost)

TN, FP, FN, TP = confusion_matrix(y_cv,y_cv_pred)

print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)

accuracy = manual_accuracy(TP,TN,y_cv)
precision = precision_score(TP,FP)
recall = recall_score(TP,FN)
f1 = f1_score(precision, recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)