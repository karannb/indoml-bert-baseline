'''
This file has several simple wrappers around sklearn evaluation functions
for the current setting.
'''
import os
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


def postprocess(results_dir: str = "outputs/") -> None:
    '''
    Simple function to postprocess saved reuslts to upload to 
    CodaLab.
    '''

    # Dictionary to store the combined data
    combined_data = {}
    # Loop over all the JSON files in the directory
    relevant_names = ["details_Brand", "L0_category", "L1_category", "L2_category", "L3_category", "L4_category"] # need to maintain order.
    for name in relevant_names:
        json_name = "results_" + name + ".json"
        with open(os.path.join(results_dir, json_name), 'r') as f:
            data = json.load(f)
            
            for item in data:
                indoml_id = item["indoml_id"]
                
                # Initialize the dictionary for this indoml_id if not already present
                if indoml_id not in combined_data:
                    combined_data[indoml_id] = {}
                
                # Merge the current dictionary's data into the combined dictionary
                combined_data[indoml_id].update(item)

    # sort by indoml_id
    combined_data = dict(sorted(combined_data.items()))

    out_file_name = "outputs/attribute_test_bert_baseline.predict" #TODO : CHANGE THE NAME!
    out_file = open(out_file_name, "w")
    for _, item in combined_data.items():
        out_file.write(json.dumps(item)+ "\n") # store as jsonl
    
    out_file.close()

    # zip the predcition for upload
    command = ['zip', 'default_submission.zip', out_file_name.split('/')[1]] #TODO : CHANGE THE NAME!
    result = subprocess.run(command, capture_output=True, cwd=os.path.dirname(out_file_name))

    if result.returncode == 0:
        print("Zipping successful.")
    else:
        print("Zipping failed.")
    
    return


if __name__ == '__main__':
    
    postprocess()
    