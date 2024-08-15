import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

import torch
from torch.utils.data import Dataset, DataLoader


class AmznReviewsDataset(Dataset):

    def __init__(self, data_dir, split="test", output="details_Brand", trim=False):

        self.data_dir = data_dir
        self.split = split
        self.output = output

        start = time()
        # Load the data
        self.data = self._load_data(data_dir, split, output)

        # Load the map
        self.out2idx = self._load_map(output)

        # Categorize the data using the map
        if "na" in self.out2idx:
            print("Removing examples with 'na' label.")
            print(f"Original data has {len(self.data)} examples.")
            self.data = self.data[self.data[output] != "na"] # remove examples with 'na' label
            self.out2idx = {k: v for k, v in self.out2idx.items() if k != "na"} # remove 'na' from the map
            
        self.data[output] = self.data[output].map(self.out2idx)

        # Create input
        # Convert all relevant columns to string
        self.data[["title", "store", "details_Manufacturer"]] = self.data[
            ["title", "store", "details_Manufacturer"]
        ].astype(str)

        # Create 'input' column
        self.data["input"] = "Product Name: " + self.data["title"]

        # Add 'store' and 'details_Manufacturer' information if they are not 'None'
        self.data.loc[self.data["store"] != None, "input"] += (
            ", Sold at store: " + self.data.loc[self.data["store"] != None, "store"]
        )
        self.data.loc[self.data["details_Manufacturer"] != None, "input"] += (
            ", Manufactured by: "
            + self.data.loc[
                self.data["details_Manufacturer"] != None, "details_Manufacturer"
            ]
        )

        # trim the dataset for debugging
        if trim:
            if split == "train":
                idx = np.random.choice(len(self.data), 50000, replace=False)
                self.data = self.data.iloc[idx]
            elif split == "val":
                idx = np.random.choice(len(self.data), 4000, replace=False)
                self.data = self.data.iloc[idx]
            else: # don't touch the test set
                pass
        end = time()

        print(f"Data loaded in {(end-start)/60:.3f} minutes.")
        print(f"{split} data has {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data.iloc[idx]

        return {"input": example["input"], "output": example[self.output]}

    def _load_data(self, data_dir, split, output) -> pd.DataFrame:
        '''
        Creates a unified dataFrame from the json files consisting of the data and labels.
        '''
        
        data = json.load(open(f"{data_dir}/{split}.json", "r"))
        labels = json.load(open(f"{data_dir}/{split}_sol.json", "r"))
        
        keys = list(data[0].keys()) + [key for key in labels[0].keys() if key == output]
        
        df = {key: [] for key in keys}
        
        for d, l in tqdm(zip(data, labels), desc=f"Loading {split} data", total=len(data)):
            for key in keys:
                val = d.get(key, l.get(key, None))
                df[key].append(val)
                
        df = pd.DataFrame(df)
        
        return df
    
    def _load_map(self, column) -> dict:
        '''
        Aux function to load the map from column to index.
        Inputs the col_name to load the map from.
        '''

        with open(f"{self.data_dir}/maps/{column}2idx.pkl", "rb") as f:
            map = pickle.load(f)

        return map


def reviewCollate(batch):

    inputs = [example["input"] for example in batch]
    outputs = torch.LongTensor([example["output"] for example in batch])

    return {"input": inputs, "output": outputs}


class AmznReviewsDataLoader(DataLoader):
    '''
    Utility class as a DataLoader.
    '''

    def __init__(self, *args, **kwargs):
        assert "collate_fn" not in kwargs, "collate_fn is not allowed"
        kwargs["collate_fn"] = reviewCollate
        super(AmznReviewsDataLoader, self).__init__(*args, **kwargs)
        

if __name__ == '__main__':
    
    dataset = AmznReviewsDataset(data_dir="data/", split="train", output="details_Brand")
    dataloader = AmznReviewsDataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch in dataloader:
        print(batch)
        break
