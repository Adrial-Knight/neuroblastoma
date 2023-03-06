from torch import nn
from torchvision import models

class ImprovedResNet(nn.Module):
    def __init__(self, resnet_version=18):
        super(ImprovedResNet, self).__init__()

        if resnet_version == 18:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif resnet_version == 34:
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif resnet_version == 50:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif resnet_version == 101:
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        elif resnet_version == 152:
            self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"ResNet version {resnet_version} does not exist.")

        self.resnet.fc = nn.Sequential()
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def fine_tune(self):
        for param in self.parameters():
            param.requires_grad = True

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.fc.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())
