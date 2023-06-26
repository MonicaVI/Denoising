import torch
import torch.nn as nn
# 定义对比扩散损失函数
class ContrastiveDiffusionLoss(nn.Module):
    def __init__(self, beta):
        super(ContrastiveDiffusionLoss, self).__init__()
        self.beta = beta

    def forward(self, true, pred):
        error = torch.abs(true - pred)
        loss = torch.mean(self.beta * error - torch.log(1 - torch.exp(-self.beta * error)))
        return loss