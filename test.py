"""
    测试python语法
"""
from Util import DataUtil
import numpy as np

x = []
# 将编码设置为utf-8以便读入中文等特殊字符
path = "C:\Program Files\Git\MachineLearning\_Data\\bank1.0.txt"
with open(path, "r", encoding="utf8") as file:
    # 如果是气球数据集的话，直接以逗号分隔数据即可
    for sample in file:
        x.append(sample.strip().split(","))
x = np.array(x).T
whether_continuous = [False] * 17
continuous_lst = [0, 5, 9, 11, 12, 13, 14, 16]
for cl in continuous_lst:
    whether_continuous[cl] = True
x = x[np.array(whether_continuous)].T
file = open("C:\Program Files\Git\MachineLearning\_Data\\bank2.0.txt", "w")
for i, sample in enumerate(x):
    for j , con in enumerate(sample):
        file.write(con + ',')
    file.write("\r\n")
file.flush()
file.close()
# print(np.sum(features))