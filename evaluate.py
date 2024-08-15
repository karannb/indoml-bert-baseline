from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro', zero_division=1.0)

def get_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro', zero_division=1.0)

def get_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=1.0)