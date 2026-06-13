def confusion_matrix(y, y_pred):
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(y.shape[0]):
        if(y[i] == 0 and y_pred[i] == 0):
            TN += 1
        elif(y[i] == 1 and y_pred[i] == 0):
            FN += 1
        elif(y[i] == 1 and y_pred[i] == 1):
            TP += 1
        else:
            FP += 1
    return TN, FP, FN, TP

def manual_accuracy(TP,TN,y_pred):
    return (TP+TN)/len(y_pred)

def precision_score(TP,FP):
    return TP/(TP+FP) 

def recall_score(TP,FN):
    return TP/(TP+FN)

def f1_score(recall_sc = float, precision_sc = float):
    return 2*recall_sc*precision_sc/(precision_sc + recall_sc)