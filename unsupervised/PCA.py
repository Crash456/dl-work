import numpy as np
import matplotlib.pyplot as plt

# 1. 生成数据 (二维点云)
np.random.seed(0)
X = np.random.randn(100, 2) @ np.array([[3, 1], [1, 0.5]])  # 拉伸 + 旋转

# 2. 零均值化
X_mean = X - np.mean(X, axis=0)

# 3. 协方差矩阵
cov = np.cov(X_mean, rowvar=False)

# 4. 特征分解
eig_vals, eig_vecs = np.linalg.eigh(cov)
idx = np.argsort(eig_vals)[::-1]
eig_vecs = eig_vecs[:, idx]

# 5. 投影到第一主成分方向
pc1 = eig_vecs[:, 0]
X_proj = (X_mean @ pc1)[:, None] * pc1[None, :]  # 点云投影到 PC1 上
X_proj += np.mean(X, axis=0)  # 平移回原均值

# 6. 可视化
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="原始数据")
plt.plot(X_proj[:, 0], X_proj[:, 1], "r.", alpha=0.6, label="PC1 投影")
origin = np.mean(X, axis=0)
plt.quiver(*origin, *pc1, color="r", scale=3, label="PC1方向")
plt.legend()
plt.title("PCA 投影示意")
plt.show()
