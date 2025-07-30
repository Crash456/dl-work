# 导入数据
import numpy as np

np.random.seed(0)  # 设置随机数种子，保证每次运行结果一致，可复现性非常重要
X_train_fpath = "./data/X_train"
Y_train_fpath = "./data/Y_train"
X_test_fpath = "./data/X_test"
output_fpath = "./output_{}.csv"

# 把csv文件转换成numpy的数组
with open(X_train_fpath) as f:
    next(f)  # 跳过表头
    X_train = np.array(
        [line.strip("\n").split(",")[1:] for line in f], dtype=float
    )  # 跳过第一列，转换为二维float数组用于后续计算
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip("\n").split(",")[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip("\n").split(",")[1:] for line in f], dtype=float)


# 标准化
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])  # 如果没有指定列，则对所有列进行标准化
    if train:  # 如果是训练数据
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)  # 每一列的均值
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)  # 每一列的标准差

    # 执行标准化，且避免除以0
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std


# 标准化训练数据和测试数据
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(
    X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std
)
# 用 _ 这个变量来存储函数返回的无用值

train_size = X_train.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print("Size of training set: {}".format(train_size))
print("Size of testing set: {}".format(test_size))
print("Dimension of data: {}".format(data_dim))


def _sigmoid(z):  # sigmoid函数 将线性输出映射到(0,1)，作为概率。
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):  # 线性组合 + Sigmoid 得到预测概率。
    return _sigmoid(np.matmul(X, w) + b)


def _predict(X, w, b):  # 概率大于0.5为1，小于0.5为0，得到二分类预测结果
    return np.round(_f(X, w, b)).astype(int)


def _accuracy(Y_pred, Y_label):  # 计算准确率
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


# 均值和协方差
# 分别计算类别0和类别1的均值
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis=0)
mean_1 = np.mean(X_train_1, axis=0)

# 分别计算类别0和类别1的协方差
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# 共享协方差 = 独立的协方差的加权求和
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (
    X_train_0.shape[0] + X_train_1.shape[0]
)


# 计算协方差矩阵的逆
# 协方差矩阵可能是奇异矩阵, 直接使用np.linalg.inv() 可能会产生错误
# 通过SVD矩阵分解，可以快速准确地获得方差矩阵的逆
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# 计算w和b
w = np.dot(inv, mean_0 - mean_1)
b = (
    (-0.5) * np.dot(mean_0, np.dot(inv, mean_0))
    + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])
)

# 计算训练集上的准确率
Y_train_pred = 1 - _predict(X_train, w, b)
print("Training accuracy: {}".format(_accuracy(Y_train_pred, Y_train)))


# 预测测试集的label
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath.format("generative"), "w") as f:
    f.write("id,label\n")
    for i, label in enumerate(predictions):
        f.write("{},{}\n".format(i, label))

# 输出最重要的10个特征
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip("\n").split(",")
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
