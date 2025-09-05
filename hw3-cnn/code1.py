# 全部代码
# 导入需要的库
import os  # 处理文件和目录路径（例如加载图片路径、创建保存结果的文件夹）
import numpy as np
import cv2  # OpenCV 库，处理图像（读取、缩放、颜色转换等）
import torch
import torch.nn as nn  # 神经网络模块（卷积层、全连接层、激活函数等）
import torchvision.transforms as transforms  # 提供常用图像预处理（数据增强、归一化、裁剪等）
import pandas as pd
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # Dataset：定义自己的数据集类，告诉 PyTorch 如何加载图片和标签。

# DataLoader：批量加载数据，支持多线程加速。
import time  # 记录训练时间、推理时间


# 读取图片
# 定义一个读取图片的函数readfile()
def readfile(path, label):  # label 是一个布尔值，代表需不需要返回 y 值
    image_dir = sorted(os.listdir(path))  # 列出 path 目录下所有文件名，并按字典序排序。
    # x存储图片，每张彩色图片都是128(高)*128(宽)*3(彩色三通道)
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    # y存储标签，每个y大小为1
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(
            os.path.join(path, file)
        )  # 读取一张图片，返回 BGR 格式的 NumPy 数组。
        # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽)
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])  # 取第一个编号并转为整数，作为类别标签。
    if label:
        return x, y
    else:
        return x


# 分别将 training set、validation set、testing set 用函数 readfile() 读进来
workspace_dir = "/kaggle/input/cnn-dog/food-11"
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))


# 定义数据集
# training 时，通过随机旋转、水平翻转图片来进行数据增强（data augmentation）
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # 把 NumPy 数组（H×W×C，dtype=uint8）转成 PIL.Image 对象
        transforms.RandomHorizontalFlip(),  # 随机翻转图片
        transforms.RandomRotation(15),  # 随机旋转图片
        transforms.ToTensor(),  # 将图片变成 Tensor，并且把数值normalize到[0,1]
    ]
)
# testing 时，不需要进行数据增强（data augmentation）
test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):  # transform 用来数据增强的
        self.x = x
        # label 需要是 LongTensor 型 （在Pytorch中，主要原因是损失函数的要求）
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):  # 告诉 PyTorch 数据集有多少样本
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]  # 取出一张图片
        if self.transform is not None:
            X = self.transform(X)  # 数据增强
        if self.y is not None:
            Y = self.y[index]  # 取出对应标签
            return X, Y  # 返回（图像，标签）
        else:
            return X  # 测试集只有图像


# 训练、验证：返回（图像，标签）
# 测试：返回 （图像tensor）


# DataLoader 封装（批量加载）
batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)  # 每轮打乱顺序
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


#### 定义模型  ： 先卷积池化缩小图像，再全连接输出分类结果的标准CNN模型，最后得到11类的预测


# 先是一个卷积神经网络，再是一个全连接的前向传播神经网络。
# 卷积神经网络的一级卷积层由卷积层cov+批标准化batchnorm+激活函数ReLU+最大池化MaxPool构成。
class Classifier(nn.Module):  # Classifier 继承 nn.Module，是标准 PyTorch 模型定义方式
    def __init__(self):  # __init__ 定义网络层结构，forward 定义前向传播过程。
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                3, 64, 3, 1, 1
            ),  # 输出[64, 128, 128]             输入通道=3（RGB） 输出通道=64  卷积核 3x3  stride = 1 padding = 1
            nn.BatchNorm2d(64),  # 对每个通道做归一化，加速收敛、稳定训练。
            nn.ReLU(),
            nn.MaxPool2d(
                2, 2, 0
            ),  # 输出[64, 64, 64]  2×2 池化，stride=2 → 尺寸减半 [64, 64, 64]
            nn.Conv2d(
                64, 128, 3, 1, 1
            ),  # 输出[128, 64, 64]  卷积提取更高层次特征   池化继续减小空间尺寸，同时保留重要特征
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 输出[128, 32, 32]
            nn.Conv2d(
                128, 256, 3, 1, 1
            ),  # 输出[256, 32, 32]  特征图数量增加 → 提取更丰富的特征
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 输出[256, 16, 16]  空间尺寸减半 → 16×16
            nn.Conv2d(
                256, 512, 3, 1, 1
            ),  # 输出[512, 16, 16]    继续增加通道 → 表示“语义”更丰富 最后池化到 4×4，准备进入全连接层
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 输出[512, 8, 8]
            nn.Conv2d(512, 512, 3, 1, 1),  # 输出[512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 输出[512, 4, 4]
            # 五层卷积，最后每张图片被压缩成一个512个通道、每个通道4x4的小图
        )

        # 全连接的前向传播神经网络
        # nn.Linear(51244, 1024)：把卷积特征摊平成 1D 向量
        # 两层隐藏层（1024 → 512），ReLU 激活 → 提高非线性表达能力
        # nn.Linear(512, 11)：输出 11 个类别的 logits，用于分类（训练时通常接 Softmax/交叉熵）

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),  # 降维到1024
            nn.ReLU(),  # 非线性
            nn.Linear(1024, 512),  # 降维到512
            nn.ReLU(),
            nn.Linear(512, 11),  # 最后输出11个分类
        )

    # 向前传播（forward）
    def forward(self, x):
        out = self.cnn(x)  # 卷积特征提取
        out = out.view(
            out.size()[0], -1
        )  # 现在 [512, 4, 4] 要摊平成一维向量才能输入全连接层，[batch_ize,512*4*4]
        return self.fc(out)  # 全连接分类  过全连接层


# 训练 ： Pytorch 里最典型的训练循环，逐epoch的用训练集训练模型，用验证集评估模型
# 使用训练集training set进行训练，并使用验证集validation set来选择最好的参数。
# 如果遇到 CUDA out of memory 的报错，请尝试调小上面的batch_size = 128（比如改成64、32、16、8等，但是模型结果可能会受到部分影响）。


# 先定义训练要素
model = Classifier().cuda()  # 把模型放在GPU上
loss = nn.CrossEntropyLoss()  # 因为是分类任务，所以使用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
num_epoch = 30  # 训练30轮


# 这就是一个batch的训练流程：1 取数据  2 向前传播算预测   3 算损失   4 反向传播求梯度   5 更新参数
for epoch in range(num_epoch):
    epoch_start_time = time.time()  # 用于记录每轮训练时间
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 切换到训练模式（启用dropout/barchnorm的训练行为）
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 清空上一次的梯度
        train_pred = model(data[0].cuda())  # 向前传播，算出预测
        batch_loss = loss(
            train_pred, data[1].cuda()
        )  # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
        optimizer.step()  # 用梯度更新参数
        # 找到预测类别（概率最大的那个）
        train_acc += np.sum(
            np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
        )
        train_loss += batch_loss.item()

    # 验证集val
    model.eval()  # 切换到评估模式（禁止使用dropout、固定batchnorm均值方差）
    with torch.no_grad():  # 不算梯度，节省内存和加速
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(
                np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
            )
            val_loss += batch_loss.item()

        # 将结果 print 出來
        print(
            "[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f"
            % (
                epoch + 1,
                num_epoch,
                time.time() - epoch_start_time,
                train_acc / train_set.__len__(),
                train_loss / train_set.__len__(),
                val_acc / val_set.__len__(),
                val_loss / val_set.__len__(),
            )
        )


# 调整好参数后，使用train和validation一起训练
train_val_x = np.concatenate((train_x, val_x), axis=0)  # 将train_x和val_x拼接起来
train_val_y = np.concatenate((train_y, val_y), axis=0)  # 将train_y和val_y拼接起来
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()  # cuda加速
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(
            np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
        )
        train_loss += batch_loss.item()

        # 将结果 print 出來
    print(
        "[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f"
        % (
            epoch + 1,
            num_epoch,
            time.time() - epoch_start_time,
            train_acc / train_val_set.__len__(),
            train_loss / train_val_set.__len__(),
        )
    )


# 测试： 用刚才训练好的模型，在test set上进行测试
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        # 预测值中概率最大的下标即为模型预测的食物标签
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# 将预测结果写入 csv
with open("predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, y in enumerate(prediction):
        f.write("{},{}\n".format(i, y))
