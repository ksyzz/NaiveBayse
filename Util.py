import numpy as np
class DataUtil:
    # 定义一个方法使其能从文件中读取数据
    # 该方法接受五个参数：数据集的名字，数据集的路径，训练样本数，类别所在列，是否打乱数据
    @staticmethod
    def get_dataset(name, path, train_num=None, tar_idx=None, shuffile=False):
        x = []
        # 将编码设置为utf-8以便读入中文等特殊字符
        with open(path, "r", encoding="utf8") as file:
            # 如果是气球数据集的话，直接以逗号分隔数据即可
            for sample in file:
                x.append(sample.strip().split(","))
        # 默认打乱数据
        if shuffile:
            np.random.shuffle(x)
        # 默认类别在最后一列
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx) for xx in x])
        x = np.array(x)
        # 默认全都是训练样本
        if train_num is None:
            return x, y
        # 若传入了训练样本数，则依之将数据集切分为训练集和测试集
        return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])

    @staticmethod
    def quantize_data(x, y, wc, continuous_rate=0.1, separate=False):
        if isinstance(x, list):
            xt = map(list, zip(*x))
        else:
            xt = x.T
        features = [set(feat) for feat in xt]

        if wc is None:
            wc = np.array([len(feat) >= (continuous_rate * len(y)) for feat in features])
        elif not wc.all:
            wc = np.array([False] * len(xt))
        else:
            wc = np.asarray(wc) # asarray 将链表转换成数组
        feat_dics = [{_l: _i for _i, _l in enumerate(feat)} if not wc[i] else None for i, feat in enumerate(features)]

        if not separate:
            if np.all(~wc):
                dtype = np.int
            else:
                dtype = np.float32
            x = np.array([[feat_dics[i][l] if not wc[i] else l for i, l in enumerate(sample)] for sample in x], dtype=dtype)
        else:
            x = np.array([[feat_dics[i][l] if not wc[i] else l for i, l in enumerate(sample)] for sample in x], dtype=np.float32)
            x = (x[:, ~wc].astype(np.int), x[:, wc])
        label_dic = {l: i for i, l in enumerate(set(y))}
        y = np.array([label_dic[yy] for yy in y], dtype=np.int8)
        label_dic = {i: l for l, i in label_dic.items()}
        return  x, y, wc, features, feat_dics, label_dic