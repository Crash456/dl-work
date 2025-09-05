import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()  # 转为 tensor, [0,1]范围
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器: 784 -> 128 -> 64 -> 2 (压缩到二维latent)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # latent维度 = 2
        )
        # 解码器: 2 -> 64 -> 128 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # 输出范围 [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5

for epoch in range(num_epochs):
    for imgs, _ in train_loader:
        imgs = imgs.view(imgs.size(0), -1).to(device)  # 展平
        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 取一些测试图片
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

dataiter = iter(test_loader)
images, labels = next(dataiter)
images_flat = images.view(images.size(0), -1).to(device)

with torch.no_grad():
    outputs, z = model(images_flat)

# 显示原图和重建图
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(images[i].squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(outputs[i].cpu().view(28, 28), cmap="gray")
    axes[1, i].axis("off")
plt.suptitle("上排: 原图, 下排: 重建图")
plt.show()
