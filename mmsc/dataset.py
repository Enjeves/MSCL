import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, Alldata, target):
        self.data = []
        for data in Alldata:
            data.to_numpy()
            self.data.append(np.array(data))
        self.target = np.array(target)

    def __getitem__(self, index):
        data = [torch.tensor(data[index], dtype=torch.float) for data in self.data]
        label = torch.tensor(self.target[index], dtype=torch.float)
        return data, label

    def __len__(self):
        return len(self.target)

    # def to_dataframe(self):
    #     return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        self.datashape = []
        for data in self.data:
            self.datashape.append(data.shape)
        return self.datashape

