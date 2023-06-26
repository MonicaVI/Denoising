import os

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch

def train_net(k, data_path, epochs=5, batch_size=12, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    # add K_fold
    # 定义 K 折交叉验证迭代器
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
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        # optimizer = optim.Adam(net.parameters(), lr=lr)
        # 定义Loss算法
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()

        # train aad valid
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_accuracy = 0.0
        best_loss = 0.0
        for epoch in range(epochs):
            net.train()  # 设置模型为训练模式
            train_loss = 0.0
            num_correct = 0
            for images, labels in train_loader:
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                num_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader.dataset)
            # train_accuracy = num_correct / len(train_loader.dataset)
            train_losses.append(train_loss)
            # train_accuracies.append(train_accuracy)

            net.eval()  # 设置模型为评估模式
            val_loss = 0.0
            num_correct = 0
            with torch.no_grad():
                num_correct = 0
                for images, labels in val_loader:
                    images = images.to(device=device, dtype=torch.float32)
                    labels = labels.to(device=device, dtype=torch.float32)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    num_correct += (predicted == labels).sum().item()
            val_loss /= len(val_loader.dataset)
            # val_accuracy = num_correct / len(val_loader.dataset)
            val_losses.append(val_loss)
            # val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, "
                   f"train loss: {train_loss:.4f}, "
                   f"val loss: {val_loss:.4f}")
            # print(f"Epoch {epoch + 1}/{epochs}, "
            #       f"train loss: {train_loss:.4f}, train acc: {train_accuracy:.4f}, "
            #       f"val loss: {val_loss:.4f}, val acc: {val_accuracy:.4f}")
            # 保存最优模型
            # if val_accuracy > best_accuracy:
            #    best_accuracy = val_accuracy
            #    torch.save(net.state_dict(), "best_model.pth")
            if val_loss < best_loss:
                 best_loss = val_loss
                 torch.save(net.state_dict(), "best_model.pth")

        # 可视化训练结果曲线
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.show()
        out_folder = 'E:\\Project\\Denoising\\result'
        plt.savefig(out_folder, epoch + '_loss.png')
        # plt.savefig(os.path.join(out_folder, '{}_loss.png'.format(epoch)))
        plt.close()

        # plt.plot(train_accuracies, label="Train Acc")
        # plt.plot(val_accuracies, label="Val Acc")
        # plt.legend()
        # plt.show()
        # plt.savefig('acc.png')
        # plt.close()

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 指定训练集地址，开始训练
    data_path = "E:\\Project\\Denoising\\data\\train\\image"
    train_net(5, data_path)
