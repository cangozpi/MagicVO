import torch
from torch import nn

class CNN_backbone_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(6, 64, 7, 2, padding=3),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(64, 128, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(256, 256, 3, 1, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(256, 512, 3, 2, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 512, 3, 2, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 1024, 3, 2, padding=1),
            torch.nn.LeakyReLU(0.1)
        ).to(self.device)

    def forward(self, x):
        out = self.model(x)
        return out