from sklearn.svm import SVC
import numpy as np


def create_model(args):
    models = {
        'none': SVM(args),
        'ovr':OVRSVM(args)
    }
    return models[args.category]


class SVM():
    def __init__(self, args):
        self.model = SVC(C=args.C, kernel=args.kernel, degree=args.degree, decision_function_shape='ovo')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class OVRSVM():
    def __init__(self, args):
        self.args = args
        self.classifiers = []

    def fit(self, X_train, y_train):
        self.classifiers.clear()
        self.class_labels = np.unique(y_train)
        for i, label in enumerate(self.class_labels):
            # 将类别i与其他类型的样本组合在一起
            other_labels = np.delete(self.class_labels, label)

            other_indices_train = np.where(np.isin(y_train, other_labels))
            y_train_binary = np.zeros_like(y_train)
            y_train_binary[other_indices_train] = 1

            # 训练二分类器
            svm = SVC(decision_function_shape='ovo', kernel=self.args.kernel, degree=self.args.degree, C=self.args.C,
                      probability=True)
            svm.fit(X_train, y_train_binary)
            self.classifiers.append(svm)


    def predict(self, X_test):
        # 对测试集进行分类
        y_pred = np.zeros(X_test.shape[0])
        confidence_scores = np.zeros(X_test.shape[0])
        confidence_scores.fill(-100)
        for i, svm in enumerate(self.classifiers):
            # 获取每个分类器的置信度分数
            confidence_cur_scores = svm.predict_log_proba(X_test)[:, 0]
            y_pred[np.where(confidence_cur_scores > confidence_scores)] = self.class_labels[i]
            confidence_scores[np.where(confidence_cur_scores > confidence_scores)] = \
                confidence_cur_scores[np.where(confidence_cur_scores > confidence_scores)]
        return y_pred
