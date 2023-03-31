from torch import nn
from torchvision.models.vgg import vgg11, VGG11_Weights, vgg13, VGG13_Weights, vgg16, VGG16_Weights, vgg19, VGG19_Weights

class ImprovedVGG(nn.Module):
    def __init__(self, version=16):
        super(ImprovedVGG, self).__init__()

        if version == 11:
            self.vgg = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        elif version == 13:
            self.vgg = vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
        elif version == 16:
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        elif version == 19:
            self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"VGG version {version} does not exist.")

        self.vgg.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
        self.vgg.classifier.append(nn.Sigmoid())

    def fine_tune(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.vgg(x)

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())
