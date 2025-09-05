corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy machine learning",
    "Deep learning is fun",
]

# 分词
sentences = [sentence.lower().split() for sentence in corpus]

words = [word for sentence in sentences for word in sentence]
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

vocab_size = len(vocab)
print("词表大小:", vocab_size)

window_size = 1
data = []

for sentence in sentences:
    indices = [word2idx[w] for w in sentence]
    for center_pos in range(len(indices)):
        center_word = indices[center_pos]
        # 上下文词
        for w in range(
            max(center_pos - window_size, 0),
            min(center_pos + window_size + 1, len(indices)),
        ):
            if w != center_pos:
                context_word = indices[w]
                data.append((center_word, context_word))

print("训练样本示例:", data[:10])

import torch
import torch.nn as nn
import torch.optim as optim

embedding_dim = 5  # latent 维度，小例子


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)  # 输入词向量
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # 输出词向量

    def forward(self, center_words):
        center_embeds = self.in_embed(center_words)  # (batch, embed_dim)
        scores = center_embeds @ self.out_embed.weight.T  # (batch, vocab_size)
        return scores


model = SkipGramModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

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

embeddings = model.in_embed.weight.data
for i, word in idx2word.items():
    print(word, embeddings[i].numpy())


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# embeddings: (vocab_size, embedding_dim)
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings.numpy())

plt.figure(figsize=(8, 6))
for i, word in idx2word.items():
    x, y = emb_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word, fontsize=12)
plt.title("Word Embedding 2D Visualization")
plt.show()
