# ==========================
# Neighbor Embedding 可视化（Mac M1 安全版，只用 PCA）
# ==========================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接保存图片
import matplotlib.pyplot as plt

# ==========================
# 1. 固定随机种子
# ==========================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ==========================
# 2. 语料和词表
# ==========================
corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy machine learning",
    "Deep learning is fun",
]

# 分词
sentences = [s.lower().split() for s in corpus]

# 构建词表
words = [w for sentence in sentences for w in sentence]
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print("词表大小:", vocab_size)

# 构建 Skip-gram 数据
window_size = 1
data = []
for sentence in sentences:
    indices = [word2idx[w] for w in sentence]
    for center_pos in range(len(indices)):
        center_word = indices[center_pos]
        for w in range(
            max(center_pos - window_size, 0),
            min(center_pos + window_size + 1, len(indices)),
        ):
            if w != center_pos:
                context_word = indices[w]
                data.append((center_word, context_word))

print("训练样本示例:", data[:10])

# ==========================
# 3. Skip-gram 模型
# ==========================
embedding_dim = 5


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words):
        center_embeds = self.in_embed(center_words)
        scores = center_embeds @ self.out_embed.weight.T
        return scores


model = SkipGramModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================
# 4. 训练
# ==========================
for epoch in range(200):
    total_loss = 0
    for center, context in data:
        center = torch.tensor([center])
        context = torch.tensor([context])

        optimizer.zero_grad()
        output = model(center)
        loss = criterion(output, context)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

embeddings = model.in_embed.weight.data.numpy()

# ==========================
# 5. PCA 可视化 (Neighbor Embedding)
# ==========================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, word in idx2word.items():
    x, y = emb_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word, fontsize=12)
plt.title("Word Embedding PCA Visualization")
plt.tight_layout()
plt.savefig("neighbor_embedding_pca.png")
print("可视化图片已保存为 neighbor_embedding_pca.png")
