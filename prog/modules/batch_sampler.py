import torch
import numpy as np

#hello
class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, num_classes=10):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Divide sequences into length-based classes
        self.indices = list(range(len(data_source)))
        self.indices.sort(key=lambda x: len(data_source[x][0]))
        self.class_bins = np.array_split(self.indices, self.num_classes)

    def _batch_indices(self):
        all_batches = []
        # Shuffle indices within each bin separately
        for bin in self.class_bins:
            np.random.shuffle(bin)
            for i in range(0, len(bin), self.batch_size):
                batch = bin[i:i+self.batch_size]
                if len(batch) != self.batch_size:
                    continue
                all_batches.append(batch)
        np.random.shuffle(all_batches)

        # Flatten the list of shuffled bins
        all_indices = [batch for batches in all_batches for batch in batches]

        # Yield batches from the shuffled indices
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i+self.batch_size]
            if len(batch) == self.batch_size:  # Only yield complete batches
                yield batch

    def __iter__(self):
        return iter(self._batch_indices())

    def __len__(self):
        return len(self.data_source) // self.batch_size