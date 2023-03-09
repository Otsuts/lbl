import os
import argparse
from os import walk
import numpy as np
from models import create_model
from utils import import_forder
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='external', choices=['internal', 'external'], type=str)
    parser.add_argument('--category', default='none', choices=['none', 'ovo', 'ovr'], type=str)
    parser.add_argument('--C', default=200, type=int)
    parser.add_argument('--kernel', default='rbf', type=str)
    parser.add_argument('--degree', default=3, type=int)
    parser.add_argument('--dataset_path', default='./SEED-IV')
    parser.add_argument('--scale', default=False, type=bool)

    return parser.parse_args()


class TrainSVM():
    def __init__(self, args):
        self.args = args
        self.precision_all = 0.0
        self.model_num = 0
        self.svm = create_model(self.args)

    def train(self, method, path):
        if method == 'internal':
            for folder, subfolder_list, npy_files in walk(path):
                if npy_files:
                    # 导入文件目录
                    test_data_path = os.path.join(folder, 'test_data.npy')
                    test_data_label_path = os.path.join(folder, 'test_label.npy')
                    train_data_path = os.path.join(folder, 'train_data.npy')
                    train_data_label_path = os.path.join(folder, 'train_label.npy')
                    # 构造训练集和测试集
                    X_train = np.load(train_data_path)  # 610*62*5——>610*(62*5)

                    X_train = X_train.reshape(X_train.shape[0], -1)
                    y_train = np.load(train_data_label_path)

                    X_test = np.load(test_data_path)
                    X_test = X_test.reshape(X_test.shape[0], -1)
                    y_test = np.load(test_data_label_path)
                    self.svm.fit(X_train, y_train)
                    y_pred = self.svm.predict(X_test)
                    self.precision_all += len(np.where(y_pred == y_test)[0]) / len(y_pred)
                    self.model_num += 1
                    print(f'{self.model_num}models trained with precision {self.precision_all / self.model_num}')

        if method == 'external':
            print('training started')
            start_time = time.time()
            for i in range(15):
                train_data, train_label, test_data, test_label = import_forder(i + 1, scale=args.scale)
                X_train = np.concatenate(train_data, axis=0)  # 610*62*5——>610*(62*5)

                # X_train = X_train.reshape(X_train.shape[0], -1)
                y_train = np.concatenate(train_label, axis=0)

                X_test = np.concatenate(test_data, axis=0)
                # X_test = X_test.reshape(X_test.shape[0], -1)
                y_test = np.concatenate(test_label, axis=0)
                self.svm.fit(X_train, y_train)
                y_pred = self.svm.predict(X_test)
                self.precision_all += len(np.where(y_pred == y_test)[0]) / len(y_pred)
                self.model_num += 1
                print(f'{self.model_num}models trained with precision {self.precision_all / self.model_num}')
            print(f'training finished with time{time.time() - start_time}')


def main(args):
    mysvm = TrainSVM(args)
    mysvm.train(args.method, args.dataset_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
