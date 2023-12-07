import argparse
import torch.nn as nn
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='drop rate')
    args, _ = parser.parse_known_args()
    return args


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        ### YOUR CODE HERE
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(14))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), nn.BatchNorm2d(5))
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU(), nn.Dropout(p=0.3))
        self.fc3 = nn.Linear(84, 10)
        ### END YOUR CODE

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE
        x = self.conv1(x)
        x = self.conv2(x)
        x = view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        ### END YOUR CODE
