import torch.cuda
import torch.nn as nn
import torch.optim as optim
from models import MLP, MLQP, SOTA
from dataloader import *
import argparse
import random
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['MLP', 'MLQP', 'SOTA'])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_num", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--evaluate_interval", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default='Adam')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(1)
    test_dataset = SpiralDataset('./two_spiral_test_data.txt')
    train_dataset = SpiralDataset('./two_spiral_test_data.txt')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    X = []
    y = []
    for input, target in test_dataset:
        X.append(list(input))
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    plot_data(X, y)

    # define model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'MLP':
        model = MLP(2, args.hidden_num, 1)
    elif args.model == 'MLQP':
        model = MLQP(2, args.hidden_num, 1)
    else:
        model = SOTA(2, args.hidden_num, 1)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.BCELoss()
    # evaluation
    best_test_acc = 0.0

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        correct = 0
        total = len(train_dataset)
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            predicted = outputs.squeeze() >= 0.5
            correct += (predicted == targets).sum().item()
            loss = criterion(outputs, targets.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataset)
        train_accuracy = 100 * correct / total

        if not (epoch + 1) % args.evaluate_interval:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = len(test_dataset)
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    predicted = outputs.squeeze() >= 0.5
                    correct += (predicted == targets).sum().item()
                    loss = criterion(outputs, targets.float().view(-1, 1))
                    val_loss += loss.item() * inputs.size(0)
                val_loss /= len(test_dataset)
                test_accuracy = 100 * correct / total
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy

            print(
                f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train accuracy:{train_accuracy:.4f} Test Loss: {val_loss:.4f},  Test Accuracy:{test_accuracy:.4f}')
    print(f'Best test accuracy:{best_test_acc:.4f}')
    visualize_boundary(X, y, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
