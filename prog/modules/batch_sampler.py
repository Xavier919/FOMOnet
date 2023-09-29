import torch
import numpy as np

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, num_classes=30):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Divide sequences into length-based classes
        self.indices = list(range(len(data_source)))
        self.indices.sort(key=lambda x: len(data_source[x][0]))
        self.class_bins = np.array_split(self.indices, self.num_classes)

    def _batch_indices(self):
        # Flatten the list of bins and shuffle
        all_indices = [idx for bin in self.class_bins for idx in bin]
        np.random.shuffle(all_indices)

        # Yield batches from the shuffled indices
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i+self.batch_size]

    def __iter__(self):
        return iter(self._batch_indices())

    def __len__(self):
        return len(self.data_source) // self.batch_size