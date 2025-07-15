import numpy as np
from utils import load_train_data, normalize, save_model

# 1. 加载训练数据
x, y = load_train_data()
x, mean_x, std_x = normalize(x)

# 2. 添加偏置项
x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)

# 3. 初始化参数
dim = x.shape[1]
w = np.zeros((dim, 1))
learning_rate = 100
iterations = 5000  # 迭代次数
adagrad = np.zeros((dim, 1))
eps = 1e-10

# 4. 梯度下降训练
for t in range(iterations):
    prediction = np.dot(x, w)
    loss = np.sqrt(np.mean((prediction - y) ** 2))
    if t % 100 == 0:
        print(f"Iteration {t}: RMSE = {loss}")
    gradient = 2 * np.dot(x.T, prediction - y)
    adagrad += gradient**2
    w -= learning_rate * gradient / (np.sqrt(adagrad) + eps)

# 5. 保存模型
save_model(w, mean_x, std_x)
print("训练完成，模型参数保存为 weight.npy、mean_x.npy、std_x.npy")
