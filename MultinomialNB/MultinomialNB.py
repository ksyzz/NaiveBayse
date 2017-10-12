"""
    离散型贝叶斯框架
"""
from Basic import *
from Util import DataUtil
import time
import matplotlib.pyplot as plt
# 进行一些设置是的matplotlib能够显示中文
from pylab import mpl

class MultinomialNB(NaiveBayes):
    def feed_data(self, x, y, sample_weight=None):
        if isinstance(x, list):
            # list 转置
            features = map(list, zip(*x))
        else:
            # 数组可直接转置
            features = x.T
        # 利用Python中内置的高级数据结构——集合，获取各个纬度的特征和类别种类
        # 为了利用bincount方法来优化算法，将所有特征从0开始数值化
        # 注意：需要将数值化过程中的转换关系记录成字典，否则无法对新数据进行判断
        features = [set(feat) for feat in features]
        feat_dics = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dic = {_l: i for i, _l in enumerate(set(y))}
        # 利用转换字典更新训练集
        x = np.array([[feat_dics[i][_l] for i, _l in enumerate(sample)] for sample in x])
        y = np.array([label_dic[yy] for yy in y])
        # 利用Numpy中的bincount方法，获得各类别的数据的个数
        cat_counter = np.bincount(y)
        # 记录各纬度特征的取值个数
        n_possibilities = [len(feats) for feats in features]
        # 获取各类别数据的下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下标获取记录按类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]
        # 更新模型的各个属性
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        (self._cat_counter, self._feat_dics, self._n_possibilities) = (cat_counter, feat_dics, n_possibilities)
        self.label_dic = {i: _l for _l, i in label_dic.items()} # 反向字典
        # 调用处理样本权重的函数，以更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        # 利用Numpy的bincount方法获取带权重的条件概率的极大似然估计
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x
                ])
            else:
                self._con_counter.append([
                    np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=_p) for label, xx in self._label_zip
                ])

    # 定义核心训练函数
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        # data即为存储加了平滑项后的条件概率的数组
        data = [None] * n_dim
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [[
                (self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities) for p in range(n_possibilities)
            ] for c in range(n_category)]
        self._data = [np.array(dim_info) for dim_info in data]
        # 利用data生成决策函数
        def func(input_x, tar_category):
            rs = 1
            # 遍历各个维度， 利用data和条件独立性假设计算联合条件概率
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category][xx]
            # 利用先验概率和联合条件概率计算后验概率
            return  rs * p_category[tar_category]
        # 返回决策函数
        return func

    # 定义数值化数据的函数
    def _transfer_x(self, x):
        # 遍历每个元素，利用转换字典进行数值化
        for j, char in enumerate(x):
            x[j] = self._feat_dics[j][char]
        return x

    def visualize(self, save=False):
        # 将字体设置为仿宋
        mpl.rcParams['font.sans-serif'] = ['FangSong']
        mpl.rcParams['axes.unicode_minus'] = False
        colors = plt.cm.Paired([i / len(self.label_dic) for i in range(len(self.label_dic))])
        colors = {cat : color for cat, color in zip(self.label_dic.values(), colors)}
        # 利用转换字典定义其“反字典”，后面可视化会用
        rev_feat_dics = [{_val: _key for _key, _val in _feat_dic.items()} for _feat_dic in self._feat_dics]
        for j in range(len(self._n_possibilities)):
            rev_dic = rev_feat_dics[j]
            sj = self._n_possibilities[j]
            tmp_x = np.arange(1, sj + 1)
            title = "$j = {}, S_j = {}$".format(j + 1, sj)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dic)):
                plt.bar(tmp_x - 0.35 * c, self._data[j][c, :], width=0.35,
                        facecolor=colors[self.label_dic[c]], edgecolor="white",
                        label=u"class: {}".format(self.label_dic[c]))
            plt.xticks([i for i in range(sj + 2)], [""] + [rev_dic[i] for i in range(sj)] + [""])
            plt.ylim(0, 1.0)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))



if __name__ == '__main__':

    # 读入数据
    _x, _y = DataUtil.get_dataset("name", "C:\Program Files\Git\MachineLearning\_Data\mushroom.txt", tar_idx=0)
    # 实例化模型并进行训练、同时记录整个过程花费的时间
    learning_time = time.time()
    nb = MultinomialNB()

    nb.fit(_x, _y)
    learning_time = time.time() - learning_time
    # 评估模型的表现，同时记录评估过程花费的时间
    estimation_time = time.time()
    nb.evaluate(_x, _y)
    estimation_time = time.time() - estimation_time
    # 将记录下来的时间输出
    print(
        "Model building : {:12.6} s\n"
        "Estimation     : {:12.6} s\n"
        "Total          : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
    nb.visualize()
