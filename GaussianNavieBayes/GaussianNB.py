from Basic import *
import numpy as np
from Util import DataUtil
from MultinomialNB.MultinomialNB import MultinomialNB
from GaussianNavieBayes.GaussionFunction import GaussianFunction
class GaussianNB(NaiveBayes):
    def feed_data(self, x, y, sample_weight=None):
        # 简单的调用python自带的float方法将输入的数据数值化, sample必须为一位数组或链表
        x = np.array([list(map(lambda c:float(c), sample)) for sample in x])
        # 数值化类别向量
        labels = list(set(y))
        label_dic = {label : i for i, label in enumerate(labels)}
        y = np.array([label_dic[yy] for yy in y])
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[ci].T for ci in labels]
        # 更新模型的各个属性
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter, self.label_dic = cat_counter, {i: _l for _l, i in label_dic.items()}
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weights = sample_weight * (len(sample_weight))
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weights[label]
        #TODO 权重处理方式?

    def _fit(self, lb):
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        data = [GaussianFunction.gaussian_distribution(self._labelled_x, n_category, dim) for dim in range(len(self._x.T))]
        self._data = data
        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category](xx)
            return rs * p_category[tar_category]
        return func

    @staticmethod
    def _transfer_x(x):
        return x

if __name__ == '__main__':
    import time
     # 读入数据
    _x, _y = DataUtil.get_dataset("name", "C:\Program Files\Git\MachineLearning\_Data\mushroom.txt", tar_idx=0)
    nb = MultinomialNB()
    nb.feed_data(_x, _y)
    xs, ys = nb["x"].tolist(), nb["y"].tolist()
    train_num = 6000
    x_train, x_test = xs[:train_num], xs[train_num:]
    y_train, y_test = ys[:train_num], ys[train_num:]
    nb.fit(x_train, y_train)
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    gnb.evaluate(x_train, y_train)
    gnb.evaluate(x_test, y_test)

    # 实例化模型并进行训练、同时记录整个过程花费的时间
    # learning_time = time.time()
    #
    #
    # nb.fit(_x, _y)
    # learning_time = time.time() - learning_time
    # # 评估模型的表现，同时记录评估过程花费的时间
    # estimation_time = time.time()
    # nb.evaluate(_x, _y)
    # estimation_time = time.time() - estimation_time
    # # 将记录下来的时间输出
    # print(
    #     "Model building : {:12.6} s\n"
    #     "Estimation     : {:12.6} s\n"
    #     "Total          : {:12.6} s".format(
    #         learning_time, estimation_time,
    #         learning_time + estimation_time
    #     )
    # )