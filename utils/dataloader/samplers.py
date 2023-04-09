import torch
import numpy as np


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        self.label_to_indices = {}
        for i, label_value in enumerate(label):
            if label_value not in self.label_to_indices:
                self.label_to_indices[label_value] = []
            self.label_to_indices[label_value].append(i)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch_indices = []
            classes = np.random.choice(list(self.label_to_indices.keys()), self.n_cls, replace=False)
            for class_label in classes:
                indices = self.label_to_indices[class_label]
                batch_indices += np.random.choice(indices, self.n_per, replace=False).tolist()
            yield torch.tensor(batch_indices)
