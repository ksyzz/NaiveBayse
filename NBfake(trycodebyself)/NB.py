# 测试用
from Data import DataUtil
import numpy as np
class NaiveBayse:
    if __name__ == '__main__':
        from NB import NaiveBayse
        NB = NaiveBayse()
        NB.predict("C:\\balloon1.0.txt", "C:\\balloon1.5.txt")
    def __init__(self):
        self._x = self._y = None # 数据集合
        self._dics_x = self._dics_y = None # 数据字典
        self._possibilities = None # 先验概率
        self._func = None # 决策函数
        self._count_y = None # y取值和出现次数
        self._features_count = None # 各特征的数量
        self._condition_p = None # 条件概率


    def dealdata(self, path):
        dx, dy = DataUtil.getdata(path)
        self._dics_y = {_y : i for i, _y in enumerate(set(dy))}
        self._dics_x = [{_x : i for i, _x in enumerate(set(x))} for x in dx.T]
        self._y = [self._dics_y[y] for y in dy]
        self._x = [[self._dics_x[x_idx][x] for x_idx, x in enumerate(sample)] for sample in dx]
        self._count_y = np.bincount(self._y)
        self._possibilities = [(yy + 1) / (sum(self._count_y) + len(self._count_y)) for yy in self._count_y]
        self._features_count = [len(set(feature)) for feature in np.array(self._x).T]
        y = np.array(self._y)
        x = np.array(self._x)
        feats = [y == value for value in range(len(self._count_y))]
        print(len(x))
        label_x = [x[ci].T for ci in feats]
        self._condition_p = [[np.bincount(label_x[yy][i]) for yy in range(len(self._count_y))] for i in range(len(self._features_count))]
        condition_p = [None] * len(self._features_count)
        for dim, count in enumerate(self._features_count):
            condition_p[dim] = [[(self._condition_p[dim][k][i] + 1) / (self._count_y[k] + count * 1) for i in range(self._features_count[dim])]
                                for k in range(len(self._count_y))]
        def func(input_x):
            rs = [1] * len(self._count_y)
            for yy in range(len(self._count_y)):
                for idx, value in enumerate(input_x):
                    rs[yy] *= condition_p[idx][yy][value]
                rs[yy] *= self._possibilities[yy]
            idx = rs.index(np.max(rs))
            for k, v in self._dics_y.items():
                if v == idx:
                    return k
        return func

    def predict(self, path, predictt_path):
        func = self.dealdata(path)
        dx, dy = DataUtil.getdata(predictt_path)
        _x = [[self._dics_x[i][j] for i, j in enumerate(xx)] for xx in dx]
        x = np.array(_x)

        for i in range(len(x)):
            rs = func(x[i])
            print(rs)