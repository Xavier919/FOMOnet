from torch.utils.data import Dataset

class Transcripts(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        return (current_sample,current_target)