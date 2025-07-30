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


# 划分训练集与验证集
def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# 把数据分成训练集和验证集
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print("Size of training set: {}".format(train_size))
print("Size of development set: {}".format(dev_size))
print("Size of testing set: {}".format(test_size))
print("Dimension of data: {}".format(data_dim))


# 一些有用的函数
def _shuffle(X, Y):  # 随机打乱数据
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def _sigmoid(z):  # sigmoid函数 将线性输出映射到(0,1)，作为概率。
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):  # 线性组合 + Sigmoid 得到预测概率。
    return _sigmoid(np.matmul(X, w) + b)


def _predict(X, w, b):  # 概率大于0.5为1，小于0.5为0，得到二分类预测结果
    return np.round(_f(X, w, b)).astype(int)


def _accuracy(Y_pred, Y_label):  # 计算准确率
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


# 定义损失函数
def _cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot(
        (1 - Y_label), np.log(1 - y_pred)
    )
    return cross_entropy


# 定义梯度
def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b)  # 模型输出概率值
    pred_error = Y_label - y_pred  # 残差（预测 - 真实）
    w_grad = -np.sum(pred_error * X.T, 1)  # 权重的梯度：矩阵乘法后对每一维特征求和
    b_grad = -np.sum(pred_error)  # 偏置的梯度：所有样本的误差求和
    return w_grad, b_grad


# 训练
# 初始化权重w和b，令它们都为0
w = np.zeros((data_dim,))  # [0,0,0,...,0]
b = np.zeros((1,))  # [0]

# 训练时的超参数
max_iter = 10
batch_size = 8
learning_rate = 0.2

# 保存每个iteration的loss和accuracy，以便后续画图
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# 累计参数更新的次数
step = 1

#  Epoch 循环训练 + mini-batch 梯度下降
for epoch in range(max_iter):
    # 在每个epoch开始时，随机打散训练数据
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini-batch训练
    for idx in range(int(np.floor(train_size / batch_size))):
        # 从训练集 X_train 中取出第 idx 个 mini-batch 的数据子集。
        # 从训练集 Y_train 中取出第 idx 个 mini-batch 的数据子集。
        X = X_train[idx * batch_size : (idx + 1) * batch_size]
        Y = Y_train[idx * batch_size : (idx + 1) * batch_size]

        # 计算梯度
        w_grad, b_grad = _gradient(X, Y, w, b)

        # 梯度下降法更新
        # 学习率随时间衰减
        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad

        step = step + 1

    # 计算训练集和验证集的loss和accuracy
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print("Training loss: {}".format(train_loss[-1]))
print("Development loss: {}".format(dev_loss[-1]))
print("Training accuracy: {}".format(train_acc[-1]))
print("Development accuracy: {}".format(dev_acc[-1]))

# 画出训练集和验证集的loss和acc
import matplotlib.pyplot as plt

plt.plot(train_loss)  # 画出训练集的 loss 曲线（纵轴是 loss，横轴是 epoch）
plt.plot(dev_loss)  # 画出验证集的 loss 曲线
plt.title("Loss")
plt.legend(["train", "dev"])  # 添加图例，标明两条曲线的含义（顺序对应上面两行 plot）
plt.savefig("loss.png")  # 将当前的图保存为图片文件 loss.png
plt.show()

plt.plot(train_acc)  # 画出训练集的准确率变化曲线
plt.plot(dev_acc)  # 画出验证集的准确率变化曲线
plt.title("Accuracy")
plt.legend(["train", "dev"])  # 添加图例，表示哪条线是哪种准确率
plt.savefig("acc.png")  # 保存这张图为 acc.png 文件
plt.show()  # 显示准确率图


# 预测测试集标签
predictions = _predict(X_test, w, b)
# 保存到output_logistic.csv
with open(output_fpath.format("logistic"), "w") as f:
    f.write("id,label\n")
    for i, label in enumerate(predictions):
        f.write("{},{}\n".format(i, label))

# 输出最重要的特征和权重
# 对w的绝对值从大到小排序，输出对应的ID
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    # 读入表头（特征名）
    content = f.readline().strip("\n").split(",")
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
