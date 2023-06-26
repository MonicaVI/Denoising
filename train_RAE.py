import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model_RAE import RAE
from utils.dataset import ISBI_Loader


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0
    count = 0
    for data in train_loader:
        inputs, targets = data[0].to(device), data[1].to(device)
        inputs = inputs.reshape(1, 1, -1)
        targets = targets.reshape(1, 1, -1)
        # 预测输出并计算损失
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = count +1
        print("count {}: Train Loss {}".format(count, loss.item() * inputs.size(0)))
        train_loss += loss.item() * inputs.size(0)


    # 计算平均训练损失
    train_loss = train_loss / len(train_loader.dataset)

    return train_loss


if __name__ == '__main__':
    # 加载数据集
    dataset = ISBI_Loader("E:\\Project\\Denoising\\data\\train\\image")
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=1,
                                               shuffle=True)
    # 定义模型和损失函数
    model = RAE()
    criterion = nn.L1Loss()

    # 将模型移动到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        model = model.to(device=device, dtype=torch.float64)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 开始训练
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        print("Epoch {}: Train Loss {}".format(epoch + 1, train_loss))
