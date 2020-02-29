import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np


class EEGDataset(Dataset):

    # x_tensor:
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

        assert self.x.shape[0] == self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


class AdaptDataset(Dataset):
    def __init__(self, x_tensor, z_tensor, y_tensor, data, label):
        self.x = x_tensor
        self.z = z_tensor
        self.y = y_tensor
        self.data = data
        self.label = label

    def __getitem__(self, index):
        pos = index // (22 * self.data[0, 0, -self.data.shape[1] // 5, :].shape[0])
        sample_index = index % (22 * self.data[0, 0, -self.data.shape[1] // 5, :].shape[0])
        idx = self.z[pos]
        sub = idx[0]
        source_idx = idx[1:19]
        target_idx = idx[19:]
        data_source = self.data[sub, source_idx, :, :].flatten(start_dim=0, end_dim=1)
        data_target = self.data[sub, target_idx, :, :]
        data_valid = data_target[:, :-data_target.shape[1] // 5, :].flatten(start_dim=0, end_dim=1)
        data_target = data_target[:, :-data_target.shape[1] // 5, :].flatten(start_dim=0, end_dim=1)

        label_valid = self.label[sub, target_idx, -data_target.shape[1] // 5:].flatten()
        return self.x[index], self.y[index], data_source, data_target, data_valid[sample_index], \
               label_valid[sample_index]

    def __len__(self):
        # numOfSubjects trials samples
        return self.x.shape[0] * 22 * self.data[0, 0, -self.data.shape[1] // 5, :].shape[0]

# def my_collate_fn(batch):
