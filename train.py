from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch

def train_net(net, device, data_path, epochs=10, batch_size=16, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)

    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # 记录每个epoch的loss值
    losses = []
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 3折交叉验证
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(isbi_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                                   batch_size=batch_size,
                                                   sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                                 batch_size=batch_size,
                                                 sampler=val_subsampler)

        # 训练epochs次
        for epoch in range(epochs):
            print("epoch:", epoch)

            # 训练模式
            net.train()
            # 按照batch_size开始训练
            count = 1
            epoch_loss = 0.0
            for image, label in train_loader:
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)
                epoch_loss += loss.item() * image.size(0)

                count = count + 1

                # 更新参数
                loss.backward()
                optimizer.step()

            epoch_loss /= len(train_loader.dataset)  # 计算每个样本的平均loss
            losses.append(epoch_loss)  # 记录该epoch的loss

            print('Epoch: {} Loss/train: {:.6f}'.format(epoch, epoch_loss))

            # 在验证集上测试
            net.eval()
            val_loss = 0
            with torch.no_grad():
                for image, label in val_loader:
                    # 将数据拷贝到device中
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    # 使用网络参数，输出预测结果
                    pred = net(image)
                    # 计算loss
                    loss = criterion(pred, label)
                    val_loss += loss.item() * image.size(0)

                val_loss /= len(val_loader.dataset)  # 计算每个样本的平均loss
                print('Epoch: {} Loss/val: {:.6f}'.format(epoch, val_loss))

            # 保存loss值最小的网络参数
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), 'best_model.pth')

        # 绘制Loss曲线
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "E:\\Project\\Denoising\\data\\train\\image"
    train_net(net, device, data_path)
