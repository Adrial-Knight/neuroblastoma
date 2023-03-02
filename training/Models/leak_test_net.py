from torch import nn

class LeakTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3)),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3)),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3)),
            nn.MaxPool2d((2,2)),
            nn.Flatten()
        )
        self.nn = nn.Sequential(
            nn.Linear(9216, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.cnn(x)
        b = self.nn(a)
        return b
