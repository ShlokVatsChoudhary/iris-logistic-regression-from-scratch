import numpy as np

#importing all the functions 
from model import gradient_descent, predict_logistic, iterations_to_target, compute_cost
from visualization import plot_binary_cross_entropy_cost, plot_confusion_matrix

#importing the testing, cross-validation and training data which we have already scaled in preprocessing
from experiment_1_data import  y_train, y_cv, y_test, x_train_scaled, x_cv_scaled, x_test_scaled

#importing all the verification functions from metrics
from metrics import f1_score, recall_score, precision_score, manual_accuracy, confusion_matrix

#checking the shape of the array before running
print(x_train_scaled.shape)
print(x_cv_scaled.shape)
print(x_test_scaled.shape)
print(y_train.shape)
print(y_cv.shape)
print(y_test.shape)

#initial global variables 
W_ini = np.zeros(x_train_scaled.shape[1])
b_ini = 0
alpha = 1
num_iters = 1000

#running gradient descent to find best w
W_final, b_final, j_history = gradient_descent(
    x_train_scaled,
    y_train,
    W_ini,
    b_ini,
    num_iters,
    alpha
)

print(iterations_to_target(j_history, 0.002)) #checking if minimum threshold (0.002) is reached using current alpha

print("Final W:", W_final)
print("Final b:", b_final)

#CV and test cost after running gradient descent
cost_cv = compute_cost(x_cv_scaled,y_cv,W_final,b_final)
print("cv cost = ", cost_cv)
cost_test = compute_cost(x_test_scaled,y_test,W_final,b_final)
print("test cost = ", cost_test)

plot_binary_cross_entropy_cost(j_history, "plots\experiment_1", "training_cost_curve")

#training the model using X_train 
y_train_pred = predict_logistic(x_train_scaled, W_final, b_final)
y_cv_pred = predict_logistic(x_cv_scaled, W_final, b_final)
y_test_pred = predict_logistic(x_test_scaled, W_final, b_final)

TN, FP, FN, TP = confusion_matrix(y_cv,y_cv_pred)

print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)

accuracy = manual_accuracy(TP,TN,y_test)
precision = precision_score(TP,FP)
recall = recall_score(TP,FN)
f1 = f1_score(precision, recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

plot_confusion_matrix(TN,FP,FN,TP,class_names = ["Setosa", "non-Setosa"],title = "Cross validation Confusion Matrix — Experiment 1",save_folder="plots\experiment_1", image_name= "crossValidation_confusionMatrix")

TN, FP, FN, TP = confusion_matrix(y_test,y_test_pred)

print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)

accuracy = manual_accuracy(TP,TN,y_test)
precision = precision_score(TP,FP)
recall = recall_score(TP,FN)
f1 = f1_score(precision, recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

plot_confusion_matrix(TN,FP,FN,TP,class_names = ["Setosa", "non-Setosa"],title = "Test Confusion Matrix — Experiment 1",save_folder="plots\experiment_1", image_name= "test_confusionMatrix")