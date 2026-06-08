#here i will be creating the main logic function for my model.

import numpy as np
import copy, math

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def compute_cost(X,y,W,b):
    m = X.shape[0]
    cost = 0
    epsilon = 1e-15
    for i in range(0,m):
        z_i = np.dot(X[i],W) + b
        f_wb_i = sigmoid(z_i)
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

def gradient_descent(X,y,Wini,bini,num_iters,alpha):
    j_history = []
    w = copy.deepcopy(Wini) #deepcopy so that global w does not change
    b = bini 

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X,y,w,b)
        w = w - alpha*dj_dw #shape of w annd dj_dw is same, so automatically it will pair 0-0,1,1,2-2,3-3. 
        b = b - alpha*dj_db #scalar

        if i<100000:
            j_history.append(compute_cost(X,y,w,b))
        
        if i%math.ceil(num_iters/10) == 0:
            print(f"iteration: {i:4d} : cost {j_history[-1]}")

    return w,b,j_history

def predict_logistic(X,W,b):
    y_pred = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        temp_y = sigmoid(np.dot(X[i],W) + b)

        if(temp_y>=0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred