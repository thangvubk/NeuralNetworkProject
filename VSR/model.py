import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config
MODELS = ['SRCNN', 'ESPCN']

class ModelFactory(object):
    
    def create_model(self, model_name):
        if model_name not in MODELS:
            raise Exception('cannot find {} model'.format(model_name))
        if model_name == 'SRCNN':
            return SRCNN()
        elif model_name == 'ESPCN':
            return ESPCN()


class SRCNN(nn.Module):
    """
    Model for SRCNN

    Low Resolution -> Conv1 -> Relu -> Conv2 -> Relu -> Conv3 -> High Resulution
    
    Args:
        - C1, C2, C3: num output channels for Conv1, Conv2, and Conv3
        - F1, F2, F3: filter size
    """
    def __init__(self,
                 C1=64, C2=32, C3=1,
                 F1=9, F2=1, F3=5):
        super(SRCNN, self).__init__()
        self.offset = config.SRCNN_IMG_COMP
        self.conv1 = nn.Conv2d(1, C1, F1) # in, out, kernel
        self.conv2 = nn.Conv2d(C1, C2, F2)
        self.conv3 = nn.Conv2d(C2, C3, F3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class SRCNN_proposed(nn.Module):
    def __init__(self):
        super(SRCNN_proposed, self).__init__()
        self.offset = config.SRCNN_PROP_IMG_COMP
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 5, 1)
        self.conv3 = nn.Conv2d(5, 5, 3)
        self.conv4 = nn.Conv2d(5, 32, 1)
        self.conv5 = nn.ConvTranspose2d(32, 1, 9, stride=3)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()
        self.offset = 0
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 9, 3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


