import pandas as pd
import numpy as np


# 加载训练数据并预处理
def load_train_data(path="./train.csv"):
    data = pd.read_csv(path, encoding="big5").iloc[:, 3:]  # 删除前3列
    data[data == "NR"] = 0  # 将NR替换为0]
    raw_data = data.to_numpy()  # 转换为numpy数组
    mouth_data = {}  # 存储每个月的数据
    for mouth in range(12):
        sample = np.empty([18, 480])  # 每个月18行480列
        for day in range(20):
            # 每天18行24列
            sample[:, day * 24 : (day + 1) * 24] = raw_data[
                18 * (20 * mouth + day) : 18 * (20 * mouth + day + 1), :
            ]
        mouth_data[mouth] = sample  # 存储每个月的数据
    # 12个月，每个月471个样本，每个样本18行9列
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)  # 12个月，每个月471个样本，每个样本1个标签
    for mouth in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour >= 14:
                    continue
                # 每个样本18行9列
                x[mouth * 471 + day * 24 + hour, :] = mouth_data[mouth][
                    :, day * 24 + hour : day * 24 + hour + 9
                ].reshape(1, -1)
                # 标签为第10行的值
                y[mouth * 471 + day * 24 + hour, 0] = mouth_data[mouth][
                    9, day * 24 + hour + 9
                ]
    return x, y


# 加载测试数据
def load_test_data(path="./test.csv"):
    test_data = pd.read_csv("./test.csv", header=None, encoding="big5").iloc[:, 2:]
    test_data[test_data == "NR"] = 0
    data = test_data.to_numpy()
    # 将test数据也变成 240 个维度为 18 * 9 + 1 的数据。
    test_x = np.empty([240, 18 * 9], dtype=float)
    for i in range(240):
        test_x[i, :] = data[18 * i : 18 * (i + 1), :].reshape(1, -1)
    return test_x


# 标准化训练数据
def normalize(x):
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x_norm = (x - mean_x) / (std_x + 1e-10)
    return x_norm, mean_x, std_x


# 标准化测试集
def normalize_with(x, mean_x, std_x):
    return (x - mean_x) / (std_x + 1e-10)


# 保存模型参数
def save_model(w, mean_x, std_x):
    np.save("weight.npy", w)
    np.save("mean_x.npy", mean_x)
    np.save("std_x.npy", std_x)


# 读取模型参数
def load_model():
    w = np.load("weight.npy")
    mean_x = np.load("mean_x.npy")
    std_x = np.load("std_x.npy")
    return w, mean_x, std_x
