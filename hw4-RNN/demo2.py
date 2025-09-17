import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F


# load data（分词操作，得到句子列表）
def load_training_data(path="training_label.txt"):
    if "training_label" in path:
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip("\n").split(" ") for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, "r") as f:
            lines = f.readlines()
            x = [line.strip("\n").split(" ") for line in lines]
        return x


def load_testing_data(path="testing_data.txt"):
    with open(path, "r") as f:
        lines = f.readlines()
        X = ["".join(line.strip("\n").split(",")[1:]) for line in lines[1:]]
        X = [sen.split(" ") for sen in X]
    return X


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy  # 返回的是正确预测的样本数量


# 训练word2vec模型

from gensim.models import Word2Vec


# 直接用的gensim里面的word2vec模型，来实现，没有关注底层
def train_word2vec(x):
    model = Word2Vec(
        x, vector_size=250, window=5, min_count=5, workers=12, epochs=10, sg=1
    )
    return model


train_x, y = load_training_data("training_label.txt")
train_x_no_label = load_training_data("training_nolabel.txt")
test_x = load_testing_data("testing_data.txt")

model = train_word2vec(train_x + test_x)
model.save("w2v1.model")

# 以上是把数据做成了250维度的向量  还没有做成把句子做成矩阵，记下来继续数据的预处理


class Preprocess:
    def __init__(self, sentences, sen_len, w2v_path):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len  # 每个句子固定长度为sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        if load:
            self.get_w2v_model()
        else:
            raise NotImplementedError
        for i, word in enumerate(self.embedding.wv.index_to_key):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("<pad>")
        self.add_embedding("<unk>")
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        if len(sentence) >= self.sen_len:
            sentence = sentence[: self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<pad>"])
            assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<unk>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.tensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)


from torch.utils.data import DataLoader, Dataset


class TwitterDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


from torch import nn


class LSTM_Net(nn.Module):
    def __init__(
        self,
        embedding,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout=0.5,
        fix_embedding=True,
    ):
        super(LSTM_Net, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


def traning(batch_size, n_epoch, lr, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        model.train()
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()
            accuracy = evaluation(outputs, labels)
            total_acc += accuracy / batch_size
            total_loss += batch_loss.item()
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            batch_loss = loss(outputs, labels)
            accuracy = evaluation(outputs, labels)
            total_acc += accuracy / batch_size
            total_loss += batch_loss.item()
        if total_acc > best_acc:
            best_acc = total_acc
            torch.save(model.state_dict(), "ckpt_weights1.pth")


from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sen_len = 20
fix_embedding = True
batch_size = 128
epoch = 5
lr = 0.001
w2v_path = "w2v1.model"

train_x, y = load_training_data("training_label.txt")
train_x_no_label = load_training_data("training_nolabel.txt")

preprocess = Preprocess(train_x, sen_len, w2v_path)
embedding = preprocess.make_embedding()
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

model = LSTM_Net(
    embedding,
    embedding_dim=250,
    hidden_dim=150,
    num_layers=1,
    dropout=0.5,
    fix_embedding=fix_embedding,
)
model = model.to(device)
X_train, X_val, y_train, y_val = train_test_split(
    train_x, y, test_size=0.2, random_state=42
)
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)
traning(batch_size, epoch, lr, train_loader, val_loader, model, device)


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            ret_output += outputs.int().tolist()
    return ret_output


model.load_state_dict(torch.load("ckpt_weights1.pth", map_location=device))
test_x = load_testing_data("testing_data.txt")
preprocess = Preprocess(test_x, sen_len, w2v_path)
embedding = preprocess.make_embedding()
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)
outputs = testing(batch_size, test_loader, model, device)
tmp = pd.DataFrame({"Id": [i for i in range(len(outputs))], "label": outputs})
tmp.to_csv("result.csv", index=False)
