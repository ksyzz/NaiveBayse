import numpy as np
from GaussianNavieBayes.GaussianNB import *
from MultinomialNB.MultinomialNB import *
class MergedNB(NaiveBayes):
    """
        初始化结构
        self._whether_discrete:记录各个维度的变量是否是离散型变量
        self._wheter_continuous:记录各个维度的变量是否是连续型变量
        self._multinomial,self_gaussian:离散型、连续型朴素贝叶斯模型
    """
    def __init__(self, whether_continuous=None):
        self._multinomial = MultinomialNB()
        self._gaussian = GaussianNB()
        if whether_continuous is None:
            self._whether_discrete = self._wheter_continuous = None
        else:
            self._wheter_continuous = np.array(whether_continuous)
            self._whether_discrete = ~self._wheter_continuous

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)

        x, y, wc, features, feat_dics, label_dic = DataUtil.quantize_data(x, y, wc=self._wheter_continuous, separate=True)
        if self._wheter_continuous is None:
            self._wheter_continuous = wc
            self._whether_discrete = ~wc
        (discrete_x, continuous_x) = x
        self.label_dic = label_dic
        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter
        labels = [y == value for value in range(len(cat_counter))]
        # 训练离散型朴素贝叶斯模型
        labelled_x = [discrete_x[ci].T for ci in labels]
        self._multinomial._x, self._multinomial._y = discrete_x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dics = [feat_dic for i, feat_dic in enumerate(feat_dics) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feat) for i, feat in enumerate(features) if self._whether_discrete[i]]
        self._multinomial.label_dic = label_dic
        # 训练连续型朴素贝叶斯
        labelled_x = [continuous_x[ci].T for ci in labels]
        self._gaussian._x, self._gaussian._y = continuous_x, y
        self._gaussian._labelled_x, self._gaussian._label_zip = labelled_x, labels

        self._gaussian._cat_counter, self._gaussian.label_dic = cat_counter, label_dic
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._gaussian.feed_sample_weight(sample_weight)
        self._multinomial.feed_sample_weight(sample_weight)

    def _fit(self, lb):
        self._multinomial.fit()
        self._gaussian.fit()
        p_category = self._multinomial.get_prior_probability(lb)
        discrete_func, continuous_func = self._multinomial._func, self._gaussian._func
        def func(input_x, tar_category):
            input_x = np.array(input_x)
            # 由于两个模型都乘了先验概率，所以需要除一个
            # return discrete_func(input_x[self._whether_discrete].astype(np.int),
            #                      tar_category) * continuous_func(input_x[self._wheter_continuous], tar_category) / p_category[tar_category]
            return continuous_func(input_x[self._wheter_continuous], tar_category)
        return func

    def _transfer_x(self, x):
        _feat_dics = self._multinomial._feat_dics
        idx = 0
        for i, discrete in enumerate(self._whether_discrete):
            if not discrete:
                x[i] = float(x[i])
            else:
                x[i] = _feat_dics[idx][x[i]]
            if discrete:
                idx += 1
        return x

if __name__ == '__main__':
    import time

    whether_continuous = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for cl in continuous_lst:
        whether_continuous[cl] = True

    train_num = 30000
    data_time = time.time()
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset("bank", "C:\Program Files\Git\MachineLearning\_Data\\bank1.0.txt", train_num=train_num)
    #x_train, y_train = DataUtil.get_dataset("bank", "C:\Program Files\Git\MachineLearning\_Data\\bank1.0.txt")

    data_time = time.time() - data_time
    learning_time = time.time()
    nb = MergedNB(whether_continuous=whether_continuous)
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    # nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time

    print(
        "Data cleaning   : {:12.6} s\n"
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            data_time, learning_time, estimation_time,
            data_time + learning_time + estimation_time
        )
    )

