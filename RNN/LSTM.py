import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_c):
        x = self.embed(x)  # (batch, seq_len, embed_size)
        out, (h, c) = self.lstm(x, h_c)
        out = self.fc(out)  # (batch, seq_len, vocab_size)
        return out, (h, c)


text = "hello world"
chars = list(set(text))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

seq_len = 4
data = []
target = []

for i in range(len(text) - seq_len):
    seq = text[i : i + seq_len]
    tgt = text[i + 1 : i + seq_len + 1]
    data.append([char2idx[ch] for ch in seq])
    target.append([char2idx[ch] for ch in tgt])

data = torch.tensor(data)  # (num_samples, seq_len)
target = torch.tensor(target)


embed_size = 10
hidden_size = 20
lr = 0.01
epochs = 200

model = CharLSTM(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 初始化隐藏状态 h0 和记忆单元 c0
h0 = torch.zeros(1, data.size(0), hidden_size)
c0 = torch.zeros(1, data.size(0), hidden_size)

for epoch in range(epochs):
    optimizer.zero_grad()
    out, (h, c) = model(data, (h0, c0))
    loss = criterion(out.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


input_seq = torch.tensor([[char2idx[ch] for ch in "hell"]])
h = torch.zeros(1, 1, hidden_size)
c = torch.zeros(1, 1, hidden_size)

generated = "hell"
for _ in range(10):
    out, (h, c) = model(input_seq, (h, c))
    pred = out.argmax(dim=2)[:, -1].item()
    generated += idx2char[pred]
    input_seq = torch.tensor([[pred]])

print("生成文本:", generated)
