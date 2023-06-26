import torch
import torch.nn as nn
from torchsummary import summary

# 定义RAE模型
class RAE(nn.Module):
    def __init__(self):
        super(RAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    model = RAE()
    print(model)
    # 将模型移动到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用torchsummary来查看模型结构
    summary(model, input_size=(1, 20))