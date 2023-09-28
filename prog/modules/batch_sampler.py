import torch
import numpy as np

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def _batch_indices(self, indices):
        batch = []
        max_len = 0
        min_len = float('inf')
        for idx in indices:
            seq_len = len(self.data_source[idx][0])
            if len(batch) < self.batch_size and seq_len - min_len <= 500 and max_len - seq_len <= 500:
                batch.append(idx)
                max_len = max(max_len, seq_len)
                min_len = min(min_len, seq_len)
            else:
                yield batch
                batch = [idx]
                max_len = seq_len
                min_len = seq_len
        if batch:
            yield batch

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        np.random.shuffle(indices)  
        indices.sort(key=lambda x: len(self.data_source[x][0]))  
        return iter(self._batch_indices(indices))

    def __len__(self):
        return len(self.data_source) // self.batch_size