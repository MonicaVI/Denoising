import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision.transforms import transforms

# 自定义含噪声的CSV文件路径
noisy_csv_file = 'E:\Project\Denoising\data\train\image'

# 自定义不含噪声的CSV文件路径
clean_csv_file = 'path_to_clean_csv.csv'

# 自定义测试数据的CSV文件路径
test_csv_file = 'path_to_test_csv.csv'

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, noisy_csv_file, clean_csv_file, transform=None):
        self.noisy_data = pd.read_csv(noisy_csv_file, header=None)
        self.clean_data = pd.read_csv(clean_csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        noisy_img = self.noisy_data.iloc[idx, :].values.reshape((32, 32))
        clean_img = self.clean_data.iloc[idx, :].values.reshape((32, 32))

        # 数据预处理，可以根据需要进行调整
        noisy_img = (noisy_img - np.mean(noisy_img)) / np.std(noisy_img)
        clean_img = (clean_img - np.mean(clean_img)) / np.std(clean_img)

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img


# 加载训练数据集
train_dataset = CustomDataset(noisy_csv_file, clean_csv_file, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加载测试数据集
test_dataset = CustomDataset(test_csv_file, clean_csv_file, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 定义残差非局部均值模块
class ResidualNonLocalMean(nn.Module):
    def __init__(self, in_channels):
        super(ResidualNonLocalMean, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out

# 定义残差非局部均值噪声去除神经网络模型
class ResidualNonLocalMeanDenoiser(nn.Module):
    def __init__(self, in_channels, num_channels):
        super(ResidualNonLocalMeanDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_nonlocal1 = ResidualNonLocalMean(num_channels)
        self.res_nonlocal2 = ResidualNonLocalMean(num_channels)
        self.conv2 = nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.res_nonlocal1(out)
        out = self.res_nonlocal2(out)
        out = self.conv2(out)
        return out

# 创建模型实例
model = ResidualNonLocalMeanDenoiser(in_channels=1, num_channels=64)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, (noisy_imgs, clean_imgs) in enumerate(train_dataloader):
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        optimizer.zero_grad()

        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

# # 测试模型
# model.eval()
# model.to(device)
#
# with torch.no_grad():
#     for i, (test_img, _) in enumerate(test_dataloader):
#         test_img = test_img.to(device)
#
#         output = model(test_img)
#
#         # 这里可以对输出进行后处理或保存
