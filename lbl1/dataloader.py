import torch
from torch.utils.data import Dataset, DataLoader


class SpiralDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                # 解析每行数据
                x, y, label = line.strip().split()
                x, y = float(x), float(y)
                label = int(label)
                # 将数据转换为 Tensor 对象并添加到样本列表中
                self.samples.append((torch.tensor([x, y]), torch.tensor(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
