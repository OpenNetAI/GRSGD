import torch.nn as nn
'''
modified to fit dataset size
'''
__all__ = ['AlexNet', 'alexnet']
num_classes = 10

# 定义Alexnet网路结构
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 修改了这个地方，不知道为什么就对了
            # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(x)
        # return F.log_softmax(inputs, dim=3)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model
