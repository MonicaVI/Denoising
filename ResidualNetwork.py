import torch
import torch.nn as nn

from utils.dataset import ISBI_Loader


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResidualNetwork(nn.Module):
    def __init__(self, input_shape, num_filters, num_blocks):
        super(ResidualNetwork, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = self._make_residual_blocks(num_filters, num_blocks)
        self.output_conv = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1)

    def _make_residual_blocks(self, num_filters, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(num_filters))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.residual_blocks(out)
        out = self.output_conv(out)
        return out
# 定义模型参数
input_shape = (1, 32, 32)
num_filters = 64
num_blocks = 4
# 选择设备，有cuda用cuda，没有就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建残差网络模型
model = ResidualNetwork(input_shape, num_filters, num_blocks)
model.to(device)
model.double()
# 使用PSNR作为损失函数
def psnr_loss(y_true, y_pred):
    max_pixel = 1.0
    mse_loss = nn.MSELoss()(y_true, y_pred)
    psnr = 10.0 * torch.log10((max_pixel ** 2) / mse_loss)
    return -psnr

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = psnr_loss
# 初始化为正无穷
best_loss = float('inf')

# 指定训练集地址，开始训练
data_path = "E:\\Project\\Denoising\\data\\train\\image"
# 加载训练集
isbi_dataset = ISBI_Loader(data_path)
train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=16,
                                               shuffle=True)
# 训练模型
for epoch in range(10):
    train_loss = 0.0
    count = 1
    optimizer.zero_grad()
    # 训练模式
    model.train()
    # 按照batch_size开始训练
    for data in train_loader:
        inputs, targets = data[0].to(device), data[1].to(device)
        output = model(inputs)
        loss = loss_function(targets, output)
        count += 1
        # 保存loss值最小的网络参数
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')
        if count % 10 == 0:
            print('count：{}, Loss/train:{}'.format(count, best_loss))
        loss.backward()
        optimizer.step()