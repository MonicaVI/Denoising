from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import os
import math
import torch.nn.functional as F

# def psnr(output, label):
#     mse = torch.mean(torch.pow(output - label, 2))
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))
# def psnr_loss(output, label):
#     mse = torch.mean(torch.pow(output - label, 2))
#     psnr = 10 * torch.log10(1 / mse)
#     loss = -psnr
#     return loss
class PSNRLoss(torch.nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, img1, img2):
        mse = F.mse_loss(img1, img2)
        psnr_value = 20 * math.log10(255.0 / math.sqrt(mse.item()))
        psnr_loss = 1/psnr_value
        return psnr_loss


def train_net(device, data_path, epochs=50, batch_size=32, lr=0.0001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    # add K_fold
    k = 5
    n_train_samples = len(isbi_dataset)
    fold_size = n_train_samples // k
    indices = list(range(n_train_samples))
    k_fold_splits = []
    for kth_fold in range(k):
        start = kth_fold * fold_size
        end = start + fold_size
        if kth_fold == k - 1:  # 如果是最后一个折，则包括所有剩余样本
            end = n_train_samples
        k_indices = indices[start:end]
        k_fold_splits.append(k_indices)

    # 循环训练和验证每个折
    for kth_fold in range(k):
        print(f"Training on fold {kth_fold + 1}/{k}...")
        # 创建训练集和验证集索引
        train_indices = []
        val_indices = k_fold_splits[kth_fold]
        for i in range(k):
            if i != kth_fold:
                train_indices += k_fold_splits[i]

        # 创建数据加载器
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(isbi_dataset, batch_size=batch_size, sampler=train_sampler)
        val_sampler = SubsetRandomSampler(val_indices)
        val_loader = DataLoader(isbi_dataset, batch_size=batch_size, sampler=val_sampler)

        # 加载网络，图片单通道1，分类为1。
        net = UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到device中
        net.to(device=device)

        # 定义优化器算法
        # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # 定义Loss算法
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        # 创建PSNR Loss对象
        psnr_loss = PSNRLoss()

        # train aad valid
        train_losses = []
        val_losses = []
        # best_loss统计，初始化为正无穷
        best_loss = float('inf')
        for epoch in range(epochs):
            net.train()  # 设置模型为训练模式
            train_loss = 0.0
            # input data
            for images, labels in train_loader:
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                # 梯度清零
                optimizer.zero_grad()
                # 前馈计算
                outputs = net(images)
                # PSNR
                # train_psnr = psnr(outputs, labels)
                # print("Train PSNR: %.2f dB" % (train_psnr))
                # 计算损失
                loss = psnr_loss(outputs, labels)
                loss = torch.tensor(loss)
                loss.requires_grad = True
                # 反向传播及优化
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"train loss: {train_loss:.4f} "
                  )

            net.eval()  # 设置模型为验证模式
            val_loss = 0.0
            with torch.no_grad():
                num_correct = 0
                for images, labels in val_loader:
                    images = images.to(device=device, dtype=torch.float32)
                    labels = labels.to(device=device, dtype=torch.float32)
                    outputs = net(images)
                    # valid_psnr = psnr(outputs, labels)
                    # print("Valid PSNR: %.2f dB" % (valid_psnr))
                    loss = psnr_loss(outputs, labels)
                    loss = torch.tensor(loss)
                    loss.requires_grad = True
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"val loss: {val_loss:.4f} "
                  )

            if val_loss < best_loss:
                 best_loss = val_loss
                 torch.save(net.state_dict(), "best_model.pth")

        # 可视化训练结果曲线
        if not os.path.exists('result'):
            os.makedirs('result')
        plt.title("Fold_"+str(i)+"_Result")
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.savefig("./result/" + str(i) + "_loss.png")
        # plt.show()
        plt.close()

# if __name__ == "__main__":
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = "E:\\Project\\Denoising\\data\\train\\image"
train_net(device, data_path)
