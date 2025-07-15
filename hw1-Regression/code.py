# 1 载入train.csv
import sys
import pandas as pd
import numpy as np
import math

# 读入train.csv,繁体字以big5编码
data = pd.read_csv("./train.csv", encoding="big5")
# 显示前10行
print(data.head(10))
print(data.shape)  # （4320,27）

# 2 预处理
data = data.iloc[:, 3:]  # 删除前3列
data[data == "NR"] = 0  # 将NR替换为0]
raw_data = data.to_numpy()  # 转换为numpy数组
print(raw_data)  # 显示处理后的数据
print(raw_data.shape)  # （4320,24）

# 3 提取特征
mouth_data = {}  # 存储每个月的数据
for mouth in range(12):
    sample = np.empty([18, 480])  # 每个月18行480列
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[
            18 * (20 * mouth + day) : 18 * (20 * mouth + day + 1), :
        ]  # 每天18行24列
    mouth_data[mouth] = sample  # 存储每个月的数据
# 提取特征（2）
x = np.empty(
    [12 * 471, 18 * 9], dtype=float
)  # 12个月，每个月471个样本，每个样本18行9列
y = np.empty([12 * 471, 1], dtype=float)  # 12个月，每个月471个样本，每个样本1个标签
for mouth in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour >= 14:
                continue
            x[mouth * 471 + day * 24 + hour, :] = mouth_data[mouth][
                :, day * 24 + hour : day * 24 + hour + 9
            ].reshape(
                1, -1
            )  # 每个样本18行9列
            y[mouth * 471 + day * 24 + hour, 0] = mouth_data[mouth][
                9, day * 24 + hour + 9
            ]  # 标签为第10行的值
print(x)
print(y)

# 4 标准化
mean_x = np.mean(x, axis=0)  # 对每一列（每个特征）求均值，得到 shape=(162,)
std_x = np.std(x, axis=0)  # 对每一列求标准差，得到 shape=(162,)
for i in range(len(x)):  # 遍历所有的样本(5652行)
    for j in range(len(x[0])):  # 遍历每个特征162列
        if std_x[j] != 0:  # 避免除以0
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
print(x)  # 显示标准化后的数据

# 5 划分数据集
x_train_set = x[: math.floor(len(x) * 0.8), :]  # 前80%作为训练集
y_train_set = y[: math.floor(len(y) * 0.8), :]  # 前80%作为训练集标签
x_validation = x[math.floor(len(x) * 0.8) :, :]  # 后20%作为验证集
y_validation = y[math.floor(len(y) * 0.8) :, :]  # 后20%作为验证集标签
print("-----------------------")
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))

# 6 训练模型
x_train_set = np.concatenate(
    (np.ones([x_train_set.shape[0], 1]), x_train_set), axis=1
).astype(
    float
)  # 添加偏置项
dim = x_train_set.shape[1]  # 输入特征数
w = np.zeros([dim, 1])  # 初始化参数
learning_rate = 100  # 学习率
iter_time = 1000  # 迭代次数
adagrad = np.zeros([dim, 1])  # 用于Adagrad的累积梯度平方和
eps = 0.0000000001  # 防止除以0的常数
for t in range(iter_time):
    loss = np.sqrt(
        np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))
    )  # RMSE损失函数LOSS
    if t % 100 == 0:
        print(str(t) + ":" + str(loss))  # 每100次迭代输出一次损失函数
    gradient = 2 * np.dot(
        x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set
    )  # 梯度
    adagrad += gradient**2  # 累积梯度平方和
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)  # 更新参数
np.save("weight.npy", w)  # 保存参数w到文件weight.npy
print(w)

x_validation = np.concatenate(
    (np.ones([x_validation.shape[0], 1]), x_validation), axis=1
).astype(float)
val_pred = np.dot(x_validation, w)
val_loss = np.sqrt(np.mean(np.power(val_pred - y_validation, 2)))
print("Validation RMSE:", val_loss)

# 7 测试
# 读入测试数据test.csv
testdata = pd.read_csv("./test.csv", header=None, encoding="big5")
# 丢弃前两列，需要的是从第3列开始的数据
test_data = testdata.iloc[:, 2:]
# 把降雨为NR字符变成数字0
test_data[test_data == "NR"] = 0
# 将dataframe变成numpy数组
test_data = test_data.to_numpy()
# 将test数据也变成 240 个维度为 18 * 9 + 1 的数据。
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i : 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
print(test_x)  # 显示测试数据

# 7 测试
# 读入测试数据test.csv
testdata = pd.read_csv("./test.csv", header=None, encoding="big5")
# 丢弃前两列，需要的是从第3列开始的数据
test_data = testdata.iloc[:, 2:]
# 把降雨为NR字符变成数字0
test_data[test_data == "NR"] = 0
# 将dataframe变成numpy数组
test_data = test_data.to_numpy()
# 将test数据也变成 240 个维度为 18 * 9 + 1 的数据。
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i : 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
print("测试数据", test_x)  # 显示测试数据

w = np.load("./weight.npy")
ans_y = np.dot(test_x, w)
print("预测结果", ans_y)
