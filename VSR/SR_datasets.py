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

# default loader
def _gray_loader(path):
    image = scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(float)
    return image

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

class SR_dataset(Dataset):
    def __init__(self, root, scale=3, loader=_gray_loader):
        self.loader = loader
        self.paths = _get_img_paths(root)
        self.scale = scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        high_res = self.loader(self.paths[idx])
        high_res = mod_crop(high_res, self.scale)

        # downsample
        low_res = scipy.ndimage.interpolation.zoom(high_res, 1/self.scale,
                                                   prefilter=False)
        
        # compensate image size due to conv layers
        offset = config.SRCNN_PROP_IMG_COMP
        high_res = high_res[offset:-offset, offset:-offset]

        # append dummy color channel if needed
        if len(high_res.shape) == 2:
            low_res = low_res[:, :, np.newaxis]
            high_res = high_res[:, :, np.newaxis]
        
        # transform to tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)

        return low_res, high_res
        

class SRCNN_dataset(SR_dataset):

    def __getitem__(self, idx):
        high_res = self.loader(self.paths[idx])
        #high_res = high_res[:50, :50]
        high_res = mod_crop(high_res, self.scale)

        # bicubic interpolation
        low_res = scipy.ndimage.interpolation.zoom(high_res, 1/self.scale, 
                                                   prefilter=False)
        low_res = scipy.ndimage.interpolation.zoom(low_res, self.scale, 
                                                   prefilter=False)
        # due to non-upsampleing, high_res image size is smaller 
        # than low_res image size, so we have to un-padding high-res
        offset = config.SRCNN_IMG_COMP
        high_res = high_res[offset:-offset, offset:-offset]

        # append dummy color channel if needed
        if len(high_res.shape) == 2:
            low_res = low_res[:, :, np.newaxis]
            high_res = high_res[:, :, np.newaxis]

        # transform np image to torch tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)
        
        return low_res, high_res

class ESPCN_dataset(SR_dataset):
    
    def subpixel_shuffle(self, img):
        # convert img of shape (S*H, S*W, C) to (H, W, C*S**2)
        # just test for gray sclae
        # TODO: test for RBG image
        SH, SW, C = img.shape
        S = self.scale
        W = SW//S
        H = SH//S

        out = np.zeros(H, W, C*S**2)
        for h in range(H):
            for w in range(W):
                for c in range(C*S**2):
                    out[h, w, c] = image[h*S + c//S%S, w*S + c%S, 0]
        
        return out

    def __getitem__(self, idx):
        high_res = self.loader(self.paths[idx])
        high_res = mod_crop(high_res, self.scale)

        # downsample
        low_res = scipy.ndimage.interpolation.zoom(high_res, 1/self.scale,
                                                   prefilter=False)
        
        # compensate image size due to conv layers
        #offset = config.SRCNN_PROP_IMG_COMP
        #high_res = high_res[offset:-offset, offset:-offset]

        # append dummy color channel if needed
        if len(high_res.shape) == 2:
            low_res = low_res[:, :, np.newaxis]
            high_res = high_res[:, :, np.newaxis]

        # subpixel shuffle
        high_res = self.subpixel_shuffle(high_res)
        
        # transform to tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)

        return low_res, high_res







        
        
        



        
        
    
        
