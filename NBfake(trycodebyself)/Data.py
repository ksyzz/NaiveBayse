import numpy as np
# 读取数据
class DataUtil:
    def getdata(path):
        x = []
        for sample in open(path, "r", encoding="utf8"):
            x.append(sample.strip().split(","))
        y = np.array([xx.pop() for xx in x])
        x = np.array(x)
        return x, y
