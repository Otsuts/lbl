import os
import argparse
from os import walk
import numpy as np
import torch.nn
import random
from tqdm import tqdm
from models import CNN
from utils import import_forder
from dataloader import CNNDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

random.seed(114514)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='external', choices=['internal', 'external'], type=str)
    parser.add_argument('--dataset_path', default='./SEED-IV')
    parser.add_argument('--scale', default=True, type=bool)
    parser.add_argument('--across_scale', default=False, type=bool)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch_times', default=300, type=int)
    parser.add_argument('--test_iter', default=1, type=int)
    parser.add_argument('--convolution_method', default='1d', type=str)
    return parser.parse_args()


class TrainCNN():
    def __init__(self, args):
        self.device = torch.device('cuda:0')if torch.cuda.is_available() else torch.device('cpu')
        self.args = args
        self.precision_all = 0.0
        self.model_num = 0
        self.CNN = CNN(args.convolution_method).to(self.device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.acc = 0.0

    def train(self, method, path):
        if method == 'internal':
            scaler = StandardScaler()
            print('training started')
            start_time = time.time()
            for folder, subforder_list, npy_files in walk(path):
                if npy_files:
                    # 导入文件目录
                    self.model_num += 1
                    test_data_path = os.path.join(folder, 'test_data.npy')
                    test_data_label_path = os.path.join(folder, 'test_label.npy')
                    train_data_path = os.path.join(folder, 'train_data.npy')
                    train_data_label_path = os.path.join(folder, 'train_label.npy')
                    # 构造训练集和测试集
                    train_dataset = CNNDataset(train_data_path, train_data_label_path,
                                               convolution_method=args.convolution_method)
                    test_dataset = CNNDataset(test_data_path, test_data_label_path,
                                              convolution_method=args.convolution_method)
                    train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

                    optimizer = Adam(self.CNN.parameters(), lr=self.args.learning_rate)
                    best_test_acc = 0.0
                    # 开始训练
                    for epoch in tqdm(range(self.args.epoch_times)):
                        self.CNN.train()
                        train_loss = 0.0
                        correct = 0

                        for inputs, target in train_loader:
                            inputs = inputs.to(self.device)
                            target = target.unsqueeze(1)
                            # target是bs*1
                            optimizer.zero_grad()
                            outputs = self.CNN(inputs)  # bs*4
                            true_label = torch.zeros_like(target).tile(4)
                            for i, x in enumerate(target):
                                true_label[i][int(x.item())] = 1
                            loss = self.loss_func(true_label, outputs.cpu())
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                            pred = outputs.argmax(dim=-1).cpu()
                            correct += len(np.where(pred == target.squeeze())[0])
                        # 每个epoch计算训练损失
                        train_loss = train_loss / len(train_dataset)
                        precision = correct / len(train_dataset)
                        # print(f'train precision:{precision:.4f}, train loss:{train_loss:.4f}')
                        if (epoch + 1) % args.test_iter == 0:
                            self.CNN.eval()
                            test_loss = 0.0
                            correct = 0
                            with torch.no_grad():
                                for inputs, target in test_loader:
                                    inputs = inputs.to(self.device)
                                    target = target.unsqueeze(1)
                                    # target是bs*1
                                    outputs = self.CNN(inputs)  # bs*4
                                    true_label = torch.zeros_like(target).tile(4)
                                    for i, x in enumerate(target):
                                        true_label[i][int(x.item())] = 1
                                    loss = self.loss_func(true_label, outputs.cpu())
                                    test_loss += loss.item()
                                    pred = outputs.argmax(dim=-1).cpu()
                                    correct += len(np.where(pred == target.squeeze())[0])
                                precision = correct / len(test_dataset)
                                if precision > best_test_acc:
                                    best_test_acc = precision
                                test_loss = test_loss / len(test_dataset)
                                # print(f'test precision:{precision:.4f}, test loss:{test_loss:.4f}')
                    self.acc += best_test_acc
                    print(f'best test acc in model{self.model_num}: {best_test_acc}')
            print(self.acc / self.model_num)

        if method == 'external':
            print('training started')
            for i in range(15):
                X_train,y_train,X_test,y_test = import_forder(i+1,scale=args.scale,scale_across=args.across_scale)
                train_dataset = CNNDataset(data_path=None,label_path=None,data_numpy_array=X_train,label_numpy_array=y_train,
                                           convolution_method=args.convolution_method)
                test_dataset = CNNDataset(data_path=None,label_path=None,data_numpy_array=X_test,label_numpy_array=y_test,
                                          convolution_method=args.convolution_method)
                train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

                optimizer = Adam(self.CNN.parameters(), lr=self.args.learning_rate)
                best_test_acc = 0.0
                # 开始训练
                for epoch in tqdm(range(self.args.epoch_times)):
                    self.CNN.train()
                    train_loss = 0.0
                    correct = 0

                    for inputs, target in train_loader:
                        inputs = inputs.to(self.device)
                        target = target.unsqueeze(1)
                        # target是bs*1
                        optimizer.zero_grad()
                        outputs = self.CNN(inputs)  # bs*4
                        true_label = torch.zeros_like(target).tile(4)
                        for i, x in enumerate(target):
                            true_label[i][int(x.item())] = 1
                        loss = self.loss_func(true_label, outputs.cpu())
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        pred = outputs.argmax(dim=-1).cpu()
                        correct += len(np.where(pred == target.squeeze())[0])
                    # 每个epoch计算训练损失
                    train_loss = train_loss / len(train_dataset)
                    precision = correct / len(train_dataset)
                    # print(f'train precision:{precision:.4f}, train loss:{train_loss:.4f}')
                    if (epoch + 1) % args.test_iter == 0:
                        self.CNN.eval()
                        test_loss = 0.0
                        correct = 0
                        with torch.no_grad():
                            for inputs, target in test_loader:
                                inputs = inputs.to(self.device)
                                target = target.unsqueeze(1)
                                # target是bs*1
                                outputs = self.CNN(inputs)  # bs*4
                                true_label = torch.zeros_like(target).tile(4)
                                for i, x in enumerate(target):
                                    true_label[i][int(x.item())] = 1
                                loss = self.loss_func(true_label, outputs.cpu())
                                test_loss += loss.item()
                                pred = outputs.argmax(dim=-1).cpu()
                                correct += len(np.where(pred == target.squeeze())[0])
                            precision = correct / len(test_dataset)
                            if precision > best_test_acc:
                                best_test_acc = precision
                            test_loss = test_loss / len(test_dataset)
                            # print(f'test precision:{precision:.4f}, test loss:{test_loss:.4f}')
                self.acc += best_test_acc
                print(f'best test acc in model{self.model_num}: {best_test_acc}')
            print(self.acc / self.model_num)


def main(args):
    trainer = TrainCNN(args)
    trainer.train(args.method, args.dataset_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
