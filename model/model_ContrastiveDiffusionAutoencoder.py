import torch
import torch.nn as nn
from torchsummary import summary

class ContrastiveDiffusionAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(ContrastiveDiffusionAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU(),
            nn.Linear
            (256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    model = ContrastiveDiffusionAutoencoder(latent_dim=128)
    print(model)
    # 将模型移动到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 使用torchsummary来查看模型结构
    summary(model, input_size=(1, 1024))
