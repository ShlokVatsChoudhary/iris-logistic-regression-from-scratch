#here i will be creating the main logic function for my model.

import numpy as np
import copy, math

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1/(1 + np.exp(-Z))

def compute_cost(X,y,W,b):
    m = X.shape[0]
    cost = 0
    epsilon = 1e-15
    for i in range(0,m):
        z_i = np.dot(X[i],W) + b
        f_wb_i = sigmoid(z_i)
        f_wb_i = np.clip(f_wb_i, epsilon, 1 - epsilon)
        cost += -y[i]*np.log(f_wb_i + epsilon) - (1 - y[i])*np.log(1 - f_wb_i + epsilon)
    cost = cost/m
    return cost

def compute_gradient_logistic(X,y,W,b):
    m = X.shape[0]
    n = X.shape[1]
    dj_db = 0
    dj_dw = np.zeros(n)
    for i in range (0,m):
        z_i = np.dot(X[i],W) + b
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - y[i]
        for j in range (n): #w is an array so it requires gradient for individual w
            dj_dw[j] = dj_dw[j] + err_i*X[i,j]
        dj_db  = dj_db + err_i
    dj_db = dj_db/m
    dj_dw = dj_dw/m
    return dj_dw, dj_db

def gradient_descent(X, y, Wini, bini, num_iters, alpha):
    w = copy.deepcopy(Wini)
    b = bini

    initial_cost = compute_cost(X, y, w, b)
    j_history = [initial_cost]

    print(f"iteration: {0:4d} : cost {initial_cost}")

    for i in range(1, num_iters + 1):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        current_cost = compute_cost(X, y, w, b)
        j_history.append(current_cost)

        if i % math.ceil(num_iters / 10) == 0:
            print(f"iteration: {i:4d} : cost {current_cost}")

    return w, b, j_history

def predict_logistic(X,W,b):
    y_pred = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        temp_y = sigmoid(np.dot(X[i],W) + b)

        if(temp_y>=0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

def iterations_to_target(j_history, target):
    for i, cost in enumerate(j_history):
        if cost < target:
            return i
    return None