"""
    测试python语法
"""
import numpy as np
a = np.array([2, 1, 3, 4, 6, 7])
b = np.array([True, False, True, False, True, True])
for i, label in enumerate(b):
    print(a[label])
    print("-----")

# print(np.sum(features))