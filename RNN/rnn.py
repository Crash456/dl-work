import torch

# 一个简单字符串
text = "hello world"
chars = list(set(text))
vocab_size = len(chars)

# 字典映射
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

# 序列数据
seq_len = 4  # 每次输入4个字符
data = []
target = []

for i in range(len(text) - seq_len):
    seq = text[i : i + seq_len]
    tgt = text[i + 1 : i + seq_len + 1]  # 下一个字符作为目标
    data.append([char2idx[ch] for ch in seq])
    target.append([char2idx[ch] for ch in tgt])

data = torch.tensor(data)  # shape: (num_samples, seq_len)
target = torch.tensor(target)
print("数据示例:", data[:2], target[:2])

import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)  # (batch, seq_len, embed_size)
        out, h = self.rnn(x, h)  # out: (batch, seq_len, hidden_size)
        out = self.fc(out)  # (batch, seq_len, vocab_size)
        return out, h


# 参数
embed_size = 10
hidden_size = 20
lr = 0.01
epochs = 200

model = CharRNN(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 初始化隐藏状态
h0 = torch.zeros(1, data.size(0), hidden_size)

for epoch in range(epochs):
    optimizer.zero_grad()
    out, h = model(data, h0)
    loss = criterion(out.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 从 "hell" 开始生成
input_seq = torch.tensor([[char2idx[ch] for ch in "hell"]])
h = torch.zeros(1, 1, hidden_size)

generated = "hell"
for _ in range(10):
    out, h = model(input_seq, h)
    pred = out.argmax(dim=2)[:, -1].item()  # 取最后一个时间步
    generated += idx2char[pred]
    input_seq = torch.tensor([[pred]])

print("生成文本:", generated)
