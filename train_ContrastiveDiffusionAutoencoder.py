import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from model.loss_ContrastiveDiffusionLoss import ContrastiveDiffusionLoss
from model.model_ContrastiveDiffusionAutoencoder import ContrastiveDiffusionAutoencoder
from utils.dataset import ISBI_Loader


# 绘制损失图像
def plot_loss(train_losses):
    plt.plot(train_losses, label='Train Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 训练函数
def train(model, device, data_path, epochs=1000, batch_size=32, learning_rate=0.001, beta=0.01):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveDiffusionLoss(beta)

    # 初始化为正无穷
    best_loss = float('inf')
    train_losses = []  # 用于保存每个epoch的训练损失

    # 训练epochs次
    for epoch in range(epochs):
        train_loss = 0.0
        count = 1
        # 训练模式
        model.train()
        # 按照batch_size开始训练
        for data in train_loader:
            inputs, targets = data[0].to(device), data[1].to(device)
            inputs = inputs.reshape(1, 1, -1)
            targets = targets.reshape(1, 1, -1)
            # # 展平图像数据
            # image = image.view(image.size(0), -1)
            # 使用网络参数，输出预测结果
            pred = model(inputs)
            # # 将label重塑为与pred相同的尺寸
            # label = label.view(pred.size())
            # 计算loss
            loss = criterion(pred, targets)
            count += 1
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'best_model.pth')
            if count % 10 == 0:
                print('count：{}, Loss/train:{}'.format(count, best_loss))
            # 更新参数
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)  # 累计损失

        train_loss = train_loss / len(train_loader.dataset)  # 平均损失
        train_losses.append(train_loss)

    # 绘制损失图像
    plot_loss(train_losses)

# 绘制损失图像
# plot_loss(train_losses)

if __name__ == "__main__":
    # # 设置随机种子和设备
    # seed = 1234
    # torch.manual_seed(seed)
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    net = ContrastiveDiffusionAutoencoder(latent_dim=128)
    #将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "E:\\Project\\Denoising\\data\\train\\image"
    train(net, device, data_path)
