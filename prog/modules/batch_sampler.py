import torch
import numpy as np

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, num_classes=10):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Divide sequences into length-based classes
        self.indices = list(range(len(data_source)))
        self.indices.sort(key=lambda x: len(data_source[x][0]))
        self.class_bins = np.array_split(self.indices, self.num_classes)
        self.available_bins = list(self.class_bins)

    def _batch_indices(self):
        # While there are enough indices to form a batch
        while True:
            batch = []
            while len(batch) < self.batch_size:
                # If all bins are exhausted, reset available bins
                if not self.available_bins:
                    self.available_bins = list(self.class_bins)

                # Randomly select a bin from available bins
                chosen_bin_idx = np.random.choice(len(self.available_bins))
                chosen_bin = self.available_bins[chosen_bin_idx]
                
                # If the chosen bin has enough indices, sample without replacement
                if len(chosen_bin) >= self.batch_size - len(batch):
                    batch.extend(np.random.choice(chosen_bin, self.batch_size - len(batch), replace=False))
                # If not, sample all from the chosen bin and remove it from available bins
                else:
                    batch.extend(chosen_bin)
                    del self.available_bins[chosen_bin_idx]

            # If we've collected enough indices for a batch, yield it
            if len(batch) == self.batch_size:
                yield batch

    def __iter__(self):
        return iter(self._batch_indices())

    def __len__(self):
        return len(self.data_source) // self.batch_size





