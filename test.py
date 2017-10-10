from Util import DataUtil
from MultinomialNB import MultinomialNB
import time
for dataset in ("balloon1.0", "balloon1.5"):
    # 读入数据
    _x, _y = DataUtil.get_dataset(dataset, "C:\{}.txt".format(dataset))
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
