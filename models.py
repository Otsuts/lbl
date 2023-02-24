import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.name = 'MLP'
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class MLQP_layer(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output, bias=True)
        self.quadratic = nn.Linear(input, output, bias=True)

    def forward(self, x):
        linear = self.linear(x)
        quadratic = self.quadratic(x**2)
        return linear + quadratic


class MLQP(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.name = 'MLQP'
        self.fc1 = MLQP_layer(input, hidden)
        self.fc2 = MLQP_layer(hidden, output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class SOTA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        hidden_size = 64
        self.name = 'SOTA'
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
