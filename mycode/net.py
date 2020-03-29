import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Net(nn.Module):
    '''
    自定义的CNN网络，3个卷积层，包含batch norm。2个pool,
    3个全连接层，包含Dropout
    输入：28x28x1s
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            OrderedDict(
                [
                    # 28x28x1
                    ('conv1', nn.Conv2d(in_channels=1,
                                        out_channels=32,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2)),

                    ('relu1', nn.ReLU()),
                    ('bn1', nn.BatchNorm2d(num_features=32)),

                    # 28x28x32
                    ('conv2', nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)),

                    ('relu2', nn.ReLU()),
                    ('bn2', nn.BatchNorm2d(num_features=64)),
                    ('pool1', nn.MaxPool2d(kernel_size=2)),

                    # 14x14x64
                    ('conv3', nn.Conv2d(in_channels=64,
                                        out_channels=128,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)),

                    ('relu3', nn.ReLU()),
                    ('bn3', nn.BatchNorm2d(num_features=128)),
                    ('pool2', nn.MaxPool2d(kernel_size=2)),

                    # 7x7x128
                    ('conv4', nn.Conv2d(in_channels=128,
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)),

                    ('relu4', nn.ReLU()),
                    ('bn4', nn.BatchNorm2d(num_features=64)),
                    ('pool3', nn.MaxPool2d(kernel_size=2)),

                    # out 3x3x64

                ]
            )
        )

        self.classifier = nn.Sequential(


            OrderedDict(
                [
                    ('fc1', nn.Linear(in_features=3 * 3 * 64,
                                      out_features=128)),
                    ('dropout1', nn.Dropout2d(p=0.5)),

                    ('fc2', nn.Linear(in_features=128,
                                      out_features=64)),

                    ('dropout2', nn.Dropout2d(p=0.6)),

                    ('fc3', nn.Linear(in_features=64, out_features=10))
                ]
            )

        )

    def forward(self, x):
        out = self.feature(x)
        out = out.view(-1, 64 * 3 *3)
        out = self.classifier(out)
        return out
