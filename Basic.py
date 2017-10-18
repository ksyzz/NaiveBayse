import numpy as np
#定义朴素贝叶斯模型的基类， 方便以后扩展
class NaiveBayes:
    """
        p29 朴素贝叶斯模型基本架构的搭建
    """

    def __init__(self):
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dic = self._feat_dics = None

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    def feed_data(self, x, y, sample_weight=None):
        pass

    def feed_sample_weight(self, sample_weight=None):
        pass

    # 定义计算先验概率的函数，lb是各个估计中的平滑项lambda，默认值是1，默认采取拉普拉斯平滑
    def get_prior_probability(self, lb=1):
        return [(_c_num + lb ) / (len(self._y) + lb * len(self._cat_counter)) for _c_num in self._cat_counter]

    # 定义具有普适性的训练函数
    def fit(self, x=None, y=None, sample_weigh=None, lb=1):
        # 如果有传入x、y，那么就用传入的x、y初始化模型
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weigh)
        self._func = self._fit(lb)

    # 留下抽象类核心算法让子类实现
    def _fit(self, lb):
        pass

    # 定义预测单一样本的函数
    # 参数get_raw_result控制该函数是输出预测的类别还是输出相应后验概率，False输出类别，True输出后验概率
    def predict_one(self, x, get_raw_result=False):
        # 在进行预测之前，要先把新的输入数据数值化
        # 如果输入的是Numpy数组，要先将它转换成Python数组
        # 这是因为Python数组在数值化这个操作上要更快
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = x[:]

        # 调用相关方法进行数值化，该方法随具体模型的不同而不同
        x = self._transfer_x(x)

        m_arg, m_probability = 0, 0
        # 遍历各类别、找到能使后验概率最大化的类别
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if not get_raw_result:
            return self.label_dic[m_arg]
        return m_probability

    #定义预测多样本的函数，本质是不断调用上面定义的predict_one函数
    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])

    #定义能对新数据进行评估的方法，这里暂时以简单的输出准确率作为演示
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        print("Acc: {:12.6} %".format(100 * np.sum(y_pred == y) / len(y)))