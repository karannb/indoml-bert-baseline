'''
This file has several simple wrappers around sklearn evaluation functions
for the current setting.
'''
import json
import subprocess
from numpy import ndarray
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def get_precision(y_true: ndarray, y_pred: ndarray) -> float:
    return precision_score(y_true, y_pred, average='macro', zero_division=1.0)


def get_recall(y_true: ndarray, y_pred: ndarray) -> float:
    return recall_score(y_true, y_pred, average='macro', zero_division=1.0)


def get_f1(y_true: ndarray, y_pred: ndarray) -> float:
    return f1_score(y_true, y_pred, average='macro', zero_division=1.0)


def get_item_score(y_true: ndarray, y_pred: ndarray) -> float:
    pass


def postprocess(results: str = "results.json") -> None:
    '''
    Simple function to postprocess saved reuslts to upload to 
    CodaLab.
    '''
    
    results = json.load(open(results))
    out_file_name = "outputs/attribute_test_default.predict" # !!change this!!
    out_file = open(out_file_name, "w")
    for item in results:
        dict2str = str(item).replace("\'", "\"")
        out_file.write(dict2str+ "\n")
    
    out_file.close()
    
    # zip the predcition for upload
    command = ['zip', 'outputs/default_submission.zip', out_file_name]
    result = subprocess.run(command, capture_output=True)
    if result.returncode == 0:
        print("Zipped file, deleting the other *.predict file created...")
        command = ["rm", out_file_name]
        subprocess.run(command)
    else:
        print("Zipping failed. The *.predict is intact.")
    
    return


if __name__ == '__main__':
    
    postprocess()
    