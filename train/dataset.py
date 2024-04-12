from torch.utils.data import Dataset
import torch
import numpy as np

class ModelDataset(Dataset):
    def __init__(self, path_data, path_data_importance, context_length):
        self.context_length = context_length
        self.data = np.memmap(path_data, dtype='uint16', mode='r')
        if path_data_importance:
            self.data_importance = np.memmap(path_data_importance, dtype='float16', mode='r')
        else:
            self.data_importance = None
        self.dataset_length = len(self.data) // self.context_length
    
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        chunk = torch.from_numpy(self.data[start:end+1].astype('int64'))
        return_dict = {"input_ids": chunk[:-1], "labels": chunk[1:]}
        if self.data_importance is not None:
            chunk_importance = torch.from_numpy(self.data_importance[start:end].astype('float16'))
            return_dict["importance"] = chunk_importance

        return return_dict