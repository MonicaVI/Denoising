import torch
from torchsummary import summary

# 定义一个简单的CNN模型
class CNN(torch.nn.Module):
    def __init__(self, input_shape=(20, 1)):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.fc1 = torch.nn.Linear(64 * (input_shape[0] // 4), 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    model = CNN(input_shape=(20, 1))
    print(model)
    # 将模型移动到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用torchsummary来查看模型结构
    summary(model, input_size=(1, 20)) # 注意这里输入数据的形状是 (channels, time_steps)，即 (1, 20)
