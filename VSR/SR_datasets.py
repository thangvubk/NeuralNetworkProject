from __future__ import division
import torchvision
import torchvision.transforms as T
import os
import glob
import scipy.misc
import scipy.ndimage
import numpy as np

import config
from torch.utils.data import Dataset
DATASETS = 'SR_dataset, SRCNN_dataset, ESPCN_dataset'


def rgb2ycbcr(rgb):
    return np.dot(rgb[...,:3], [65.738/256, 129.057/256, 25.064/256]) + 16

# default loader
def _gray_loader(path):
    #image = scipy.misc.imread(path, flatten=False, mode='YCbCr')
    image = scipy.misc.imread(path)
    if len(image.shape) == 2:
        return image
    return rgb2ycbcr(image)

# util func
def _get_img_paths(root):
    paths = glob.glob(os.path.join(root, '*.bmp'))
    paths.sort()
    return paths

def mod_crop(image, scale):
    h, w = image.shape[0], image.shape[1]
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    return image[:h, :w]

class DatasetFactory(object):

    def create_dataset(self, name, roots, scale=3):
        train_root, val_root, test_root = roots
        if name == 'DCNN':
            return SRCNN_dataset(train_root, scale), SRCNN_dataset(val_root, scale), SRCNN_dataset(test_root, scale)
        elif name == 'SRCNN':
            return SRCNN_dataset(train_root, scale), SRCNN_dataset(val_root, scale), SRCNN_dataset(test_root, scale)
        elif name == 'ESPCN':
            return ESPCN_dataset(train_root, scale), ESPCN_dataset(val_root, scale), ESPCN_dataset(test_root, scale)

class SRCNN_dataset(Dataset):
    def __init__(self, root, scale=3, loader=_gray_loader):
        self.loader = loader
        high_res_root = os.path.join(root, 'high_res')
        low_res_root = os.path.join(root, 'low_res')
        self.hs_paths = _get_img_paths(high_res_root)
        self.ls_paths = _get_img_paths(low_res_root)
        self.scale = scale

    def __len__(self):
        return len(self.hs_paths)

    def __getitem__(self, idx):
        high_res = self.loader(self.hs_paths[idx])
        low_res = self.loader(self.ls_paths[idx])
        
        low_res = low_res[:, :, np.newaxis]
        high_res = high_res[:, :, np.newaxis]

        high_res = mod_crop(high_res, 3)
        low_res = mod_crop(low_res, 3)

        # transform np image to torch tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)

        # normalize
        low_res = low_res - 0.5
        high_res = high_res - 0.5
        
        return low_res, high_res


class ESPCN_dataset(SRCNN_dataset):
    
    def subpixel_deshuffle(self, img):
        # convert img of shape (S*H, S*W, C) to (H, W, C*S**2)
        SH, SW, C = img.shape
        S = self.scale
        W = SW//S
        H = SH//S

        out = np.zeros((H, W, C*S**2))
        for h in range(H):
            for w in range(W):
                for c in range(C*S**2):
                    out[h, w, c] = img[h*S + c//S%S, w*S + c%S, 0]
        
        return out
    
    def __getitem__(self, idx):
        high_res = self.loader(self.hs_paths[idx])
        low_res = self.loader(self.ls_paths[idx])
        
        low_res = low_res[:, :, np.newaxis]
        high_res = high_res[:, :, np.newaxis]

        high_res = self.subpixel_deshuffle(high_res)

        # transform np image to torch tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)


        low_res = low_res - 0.5
        high_res = high_res - 0.5
        
        return low_res, high_res








        
        
        



        
        
    
        
