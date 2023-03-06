import torch
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Flatten, Sequential, ReLU


class Network(torch.nn.Module):
    def __init__(self, conv_sizes, linear_sizes, dropout):
        super(Network, self).__init__()

        self.conv_sizes = conv_sizes
        self.linear_sizes = linear_sizes

        self.conv_layer_1 = Sequential(Conv2d(self.conv_sizes[0], self.conv_sizes[1], 3, padding = 1),
                                       ReLU(),
                                       MaxPool2d((2, 2)))
        self.conv_layer_2 = Sequential(Conv2d(self.conv_sizes[1], self.conv_sizes[2], 3, padding = 1),
                                       ReLU(),
                                       MaxPool2d((2, 2)))
        self.conv_layer_3 = Sequential(Conv2d(self.conv_sizes[2], self.conv_sizes[3], 3, padding = 1),
                                       ReLU(),
                                       MaxPool2d((2, 2)))
        self.lin_layer_1 = Sequential(Flatten(start_dim = 0),
                                      Dropout(dropout),
                                      Linear(self.linear_sizes[0], self.linear_sizes[1]),
                                      ReLU(),
                                      Dropout(dropout))
        self.lin_layer_2 = Sequential(Linear(self.linear_sizes[1], self.linear_sizes[2]))

    def forward(self, x):

        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.lin_layer_1(x)
        out = self.lin_layer_2(x)

        return out
