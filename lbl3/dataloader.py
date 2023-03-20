import torch
import numpy as np
from torch.utils.data import Dataset


def CNNDataset(data_path, label_path, data_numpy_array=None, label_numpy_array=None, convolution_method='1d'):
    if convolution_method == '1d':
        return CNNDataset1d(data_path, label_path, data_numpy_array=data_numpy_array, label_numpy_array=label_numpy_array)
    else:
        return CNNDataset2d(data_path, label_path, data_numpy_array=data_numpy_array, label_numpy_array=label_numpy_array)


class CNNDataset1d(Dataset):
    def __init__(self, data_path, label_path, data_numpy_array=None, label_numpy_array=None):
        super().__init__()
        if data_path:
            data_numpy = np.load(data_path)
            data_numpy = data_numpy.reshape(data_numpy.shape[0], -1)
            label_numpy = np.load(label_path)
        else:
            data_numpy = data_numpy_array
            label_numpy = label_numpy_array
        self.data = torch.from_numpy(data_numpy).unsqueeze(1)
        self.label = torch.from_numpy(label_numpy).float()

        print(f'Data shape: {self.data.shape}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :], self.label[index]


class CNNDataset2d(Dataset):
    def __init__(self, data_path, label_path, data_numpy_array=None, label_numpy_array=None):
        super().__init__()
        if data_path:
            data_numpy = np.load(data_path)
            label_numpy = np.load(label_path)
        else:
            data_numpy = data_numpy_array.reshape(data_numpy_array.shape[0], 62, -1)
            label_numpy = label_numpy_array

        self.data = torch.from_numpy(data_numpy).permute(0, 2, 1)
        self.label = torch.from_numpy(label_numpy).float()
        print(f'Data shape: {self.data.shape}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :], self.label[index]
