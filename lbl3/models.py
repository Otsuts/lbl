import torch
import torch.nn as nn


def CNN(convolution_method):
    if convolution_method == '1d':
        return CNN1d()
    if convolution_method == '2d':
        return CNN2d()


class CNN2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(5, 32, 3, 1, 1, dtype=float),
            nn.AvgPool1d(3, stride=2, padding=1),
            nn.BatchNorm1d(32, dtype=float)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1, dtype=float),
            nn.AvgPool1d(3, stride=2, padding=1),
            nn.BatchNorm1d(32, dtype=float)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1, dtype=float),
            nn.AvgPool1d(3, stride=2, padding=1),
            nn.BatchNorm1d(32, dtype=float)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1, dtype=float),
            nn.AvgPool1d(3, stride=2, padding=1),
            nn.BatchNorm1d(32, dtype=float)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 4, 128, dtype=float),
            nn.ReLU(),
            nn.Linear(128, 4, dtype=float),
            nn.Softmax(dim=1, )

        )

    def forward(self, X):
        # print(X.shape)
        X = self.conv1(X)
        # print(X.shape)
        X = self.conv2(X)
        # print(X.shape)
        X = self.conv3(X)
        # print(X.shape)
        X = self.conv4(X)
        # print(X.shape)
        X = self.fc(X.view(X.shape[0],-1))
        return X


class CNN1d(nn.Module):
    def __init__(self):
        super().__init__()
        # bz*310
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, dtype=float),
            nn.BatchNorm1d(32, dtype=float),
            nn.ReLU(),
            nn.AvgPool1d(3)
        )
        # bz*16*152
        self.conv2 = nn.Sequential(
            nn.Conv1d(32,32, kernel_size=3, stride=1, dtype=float),
            nn.BatchNorm1d(32, dtype=float),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3200, 128, dtype=float),
            nn.ReLU(),
            nn.Linear(128, 4, dtype=float),
            nn.Softmax(dim=1)

        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.fc(X.view(X.shape[0], -1))
        return X

if __name__ == '__main__':
    CNN1 = CNN1d()
    CNN2 = CNN2d()
    print(CNN1)
    print(CNN2)
