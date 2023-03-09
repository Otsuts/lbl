import os
from os import walk
import numpy as np
from sklearn.preprocessing import StandardScaler


def import_forder(test_num, path='./SEED-IV', scale=True):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    scaler = StandardScaler()

    for folder, subfolder_list, npy_files in walk(path):
        # print(folder, subfolder_list, npy_files)
        if npy_files:
            user_num = int(folder.split('_')[0].split('\\')[-1])
            test_data_path = os.path.join(folder, 'test_data.npy')
            test_data_label_path = os.path.join(folder, 'test_label.npy')
            train_data_path = os.path.join(folder, 'train_data.npy')
            train_data_label_path = os.path.join(folder, 'train_label.npy')
            if user_num != test_num:
                X_train = np.concatenate([np.load(train_data_path), np.load(test_data_path)], axis=0)
                X_train = X_train.reshape(X_train.shape[0], -1)
                y_train = np.concatenate([np.load(train_data_label_path), np.load(test_data_label_path)], axis=0)
                if scale: X_train = scaler.fit_transform(X_train)
                train_data.append(X_train)
                train_label.append(y_train)
            else:
                X_test = np.concatenate([np.load(train_data_path), np.load(test_data_path)], axis=0)
                X_test = X_test.reshape(X_test.shape[0], -1)
                y_test = np.concatenate([np.load(train_data_label_path), np.load(test_data_label_path)], axis=0)
                if scale: X_test = scaler.fit_transform(X_test)
                test_data.append(X_test)
                test_label.append(y_test)

    return train_data, train_label, test_data, test_label


import_forder(1)
