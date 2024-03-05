from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """ Pytorch Dataset for tabular data storing """
    def __init__(self, data, target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]