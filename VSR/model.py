import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import config
MODELS = ['VSRCNN', 'ESPCN', 'DCNN', 'VRES']

class ModelFactory(object):
    
    def create_model(self, model_name):
        if model_name not in MODELS:
            raise Exception('cannot find {} model'.format(model_name))
        if model_name == 'VSRCNN':
            return VSRCNN()
        elif model_name == 'ESPCN':
            return ESPCN()
        elif model_name == 'VDCNN':
            return VDCNN()
        elif model_name == 'VRES':
            return VRES()


class VSRCNN(nn.Module):
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
        super(VSRCNN, self).__init__()
        self.name = 'VSRCNN'
        self.offset = config.SRCNN_IMG_COMP
        self.conv1 = nn.Conv2d(1, C1, F1, padding=4) # in, out, kernel
        self.conv2 = nn.Conv2d(C1, C2, F2)
        self.conv3 = nn.Conv2d(C2, C3, F3, padding=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()
        self.name = 'ESPCN'
        self.offset = 0
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 9, 3, padding=1)
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.conv3(x)
        return x

class VDCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        self.name = 'VDCNN'
        self.conv_first = nn.Conv2d(1, 64, 3, padding=1)
        self.conv_next = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _res_layer(self, x):
        res = x
        out = F.relu(self.conv_next(x))
        out = self.conv_next(out)
        out += res
        out = F.relu(out)
        return out

    def forward(self, x):
        res = x
        out = F.relu(self.conv_first(x))
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self._res_layer(out)
        out += res
        out = self.conv_last(out)
        return out

class VRES(nn.Module):
    def __init__(self):
        super(VRES, self).__init__()
        self.name = 'VRES'
        self.conv_first = nn.Conv2d(5, 64, 3, padding=1)
        self.conv_next = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _res_layer(self, x):
        res = x
        out = F.relu(self.conv_next(x))
        out = self.conv_next(out)
        out += res
        out = F.relu(out)
        return out

    def forward(self, x):
        center = 2
        res = x[:, center, :, :]
        res = res.unsqueeze(1)
        out = F.relu(self.conv_first(x))
        out = self._res_layer(out)
        out = self._res_layer(out)
        out = self.conv_last(out)
        out += res
        return out

