"""
    正态分布相关计算类
"""
import numpy as np
from math import pi
sqrt_pi = (2 * pi) ** 0.5
class GaussianFunction:
     # 高斯分布密度函数
     @staticmethod
     def gaussian(x, miu, sigma):
         return np.exp(-(x - miu) ** 2 / 2 * sigma ** 2) / (sqrt_pi * sigma)

     # 定义极大似然估计的函数
     # 参数n_category代表y的种类数量
     # 参数dim代表feature的索引（x(dim)）
     @staticmethod
     def gaussian_distribution(labelled_x, n_category, dim):
        miu = [np.sum(labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [(np.sum((labelled_x[c][dim] - miu[c]) ** 2) / len(labelled_x[c][dim])) ** 0.5 for c in range(n_category)]
        def func(c):
            def g(x):
                return GaussianFunction.gaussian(x, miu[c], sigma[c])
            return g
        return [func(c) for c in range(n_category)]