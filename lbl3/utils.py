import os
from os import walk
import numpy as np
from sklearn.preprocessing import StandardScaler


def import_forder(test_num, path='./SEED-IV', scale=True, scale_across=True):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    scaler = StandardScaler()

    for folder, subfolder_list, npy_files in walk(path):
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
    X_train = np.concatenate(train_data, axis=0)  # 610*62*5——>610*(62*5)
    y_train = np.concatenate(train_label, axis=0)

    X_test = np.concatenate(test_data, axis=0)
    y_test = np.concatenate(test_label, axis=0)
    if scale_across:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


import_forder(1)
