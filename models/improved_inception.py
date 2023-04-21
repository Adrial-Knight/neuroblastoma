from torch import nn
from torchvision.models import inception_v3, Inception_V3_Weights

class ImprovedInception(nn.Module):
    def __init__(self, version=3):
        super(ImprovedInception, self).__init__()
        if version == 3:
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Inception version {version} not available.")

        self.inception.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def fine_tune(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

        for param in self.inception.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.inception(x)

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())
