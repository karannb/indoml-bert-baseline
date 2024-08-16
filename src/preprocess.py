'''
This file preprocesses the raw data
to be fed into a simple BERT model.
'''

import os
import re
import json
import pickle
from tqdm import tqdm
from typing import Optional


def preprocess_fn(fname: str, data_dir: Optional[str] = "data") -> None:
    '''
    Preprocess fname, which has the structure:
    {"indoml_id": <id>, "title": <title>, "store": <store>, "details_Manufacturer": ...}
    ...
    OR
    {"indoml_id": <id>, "details_Brand": <label>, "L0_category": <label>, "L1_category": <label>, "L2_category": ...}
    ...
    
    Stores data as a json file.
    
    Args
    ----
    fname : str
        The filename to preprocess.
        
    data_dir : str
        The directory where the data is stored.
    
    Returns
    -------
    None
    '''
    
    data = []
    file = f"{data_dir}/{fname}"
    if "solution" in fname:
        out_name = fname.split('.')[0].split('_')[1] + "_sol"
    else:
        out_name = fname.split('.')[0].split('_')[1]
    output_file = f"{data_dir}/{out_name}.json"
    if os.path.exists(output_file):
        print(f"{output_file} already exists, skipped.")
        return
    print(f"Preprocessing {file}, will store in {output_file} ...")
    with open(file, 'r') as f:
        for line in tqdm(f):
            datapoint = {}
            line = line.strip().strip('{}')
            pattern = r'"([^"]+)":\s?(?:(\d+)|"([^"]+)")' # match key: value or key: "value"
            attrs = re.findall(pattern, line)
            for attr in attrs:
                key = attr[0].strip().replace('\"', '') # remove " and empty spaces from key
                value = int(attr[1]) if attr[1] != '' else attr[2].strip().replace('\"', '')
                datapoint[key] = value
            data.append(datapoint)
            
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    return


def categorize(attr_name: str = "details_Brand") -> None:
    '''
    Simple function to categorize an attribute in a json file.
    Saves a dictionary mapping the attribute to an index and the reverse into 'maps'.
    
    Args
    ----
    json_name : str
        The name of the json file to categorize.
        
    attr_name : str
        The name of the column to categorize.
        
    Returns
    -------
    None
    '''
    if not os.path.exists("data/maps"):
        os.makedirs("data/maps")
    
    # get unique values
    col_vals = set()
    data = json.load(open(f"data/val_sol.json", 'r'))
    for item in data:
        col_vals.add(item[attr_name])
    
    # store as a dictionary
    col2idx = {col: idx for idx, col in enumerate(col_vals)}
    # revert the dictionary
    idx2col = {idx: col for col, idx in col2idx.items()}
    # save
    if os.path.exists(f'data/maps/{attr_name}2idx.pkl'):
        print(f"Maps for {attr_name} already exist, skipped.")
        return
    with open(f'data/maps/{attr_name}2idx.pkl', 'wb') as f:
        pickle.dump(col2idx, f)
    with open(f'data/maps/idx2{attr_name}.pkl', 'wb') as f:
        pickle.dump(idx2col, f)

    print(f'Categorized {col}.')
    return


if __name__ == '__main__':
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="The directoy to dtore data in.")
    
    args = parser.parse_args()
    
    preprocess_fn("attrebute_test.data", args.data_dir)
    preprocess_fn("attrebute_val.data", args.data_dir)
    preprocess_fn("attrebute_val.solution", args.data_dir)
    preprocess_fn("attrebute_train.data", args.data_dir)
    preprocess_fn("attrebute_train.solution", args.data_dir)
    
    for col in ['details_Brand', 'L0_category', 'L1_category', 'L2_category', 'L3_category', 'L4_category']:
        categorize(col)