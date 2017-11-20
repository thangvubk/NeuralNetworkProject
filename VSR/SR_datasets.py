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

class DataAugment(object):
    
    def __init__(self, root, aug_root='AugTrain', size=33, stride=33):
        self.root = root
        self.size = size
        self.stride = stride
        self.aug_root = aug_root
        

    def slicing_image(self, img, size=33, stride=33):
        """
        Slice inp image (low resolution) to obtain multiple sub-image of size (kernel x kernel)
        with the step being stride
        
        Args:
            - inp: input image (low resolution)
            - label: image of high resolution
            - I: filter size of to slice inp
            - L: filter size to slice label
            - stride: step size to move filter
        Return
            - sub_inputs: sub-images from inp
            - sub_labels: sub_images from label
        """
        sub_imgs = []
        h, w = img.shape[0], img.shape[1]
        
        for hh in range(0, h-size+1, stride):
            for ww in range(0, w-size+1, stride):
                sub_img = img[hh:hh+size, ww:ww+size]

                # Make channel value
                #sub_input = sub_input.reshape(I, I, 1)
                sub_imgs.append(sub_img)
        return sub_imgs

    def empty_augment_folder(self):
        if os.path.exists(self.aug_root):
            for root, _, paths in os.walk(self.aug_root):
                for path in paths:
                    os.remove(os.path.join(root, path))
        else:
            os.mkdir(self.aug_root)

        

    def augment(self):

        self.empty_augment_folder()

        sub_imgs = []
        paths = _get_img_paths(self.root)

        img_count = 0
        for path in paths:
            img = scipy.misc.imread(path)
            sub_imgs = self.slicing_image(img, self.size, self.stride)
            
            for sub_img in sub_imgs:
                scipy.misc.imsave(self.aug_root+'/%04d.bmp' %img_count, 
                                  sub_img.astype(np.uint8))
                img_count += 1
            

class DatasetFactory(object):

    def create_dataset(self, name, roots, scale=3):
        train_root, val_root, test_root = roots
        if name == 'SR_dataset':
            return SR_dataset(train_root, scale), SR_dataset(val_root, scale), SR_dataset(test_root, scale)
        elif name == 'SRCNN':
            return SRCNN_dataset(train_root, scale), SRCNN_dataset(val_root, scale), SRCNN_dataset(test_root, scale)
        elif name == 'ESPCN':
            return ESPCN_dataset(train_root, scale), ESPCN_dataset(val_root, scale), ESPCN_dataset(test_root, scale)

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
    def __init__(self, root, scale=3, loader=_gray_loader):
        self.loader = loader
        high_res_root = os.path.join(root, 'high_res')
        low_res_root = os.path.join(root, 'low_res')
        self.hs_paths = _get_img_paths(high_res_root)
        self.ls_paths = _get_img_paths(low_res_root)
        self.scale = scale

    def __len__(self):
        return len(self.hs_paths)

    def __getitem__1(self, idx):
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
        #high_res = high_res[offset:-offset, offset:-offset]

        # append dummy color channel if needed
        if len(high_res.shape) == 2:
            low_res = low_res[:, :, np.newaxis]
            high_res = high_res[:, :, np.newaxis]

        # transform np image to torch tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)
        
        return low_res, high_res

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

        low_res = low_res - 0.5
        high_res = high_res - 0.5
        
        return low_res, high_res




class ESPCN_dataset(SRCNN_dataset):
    
    def subpixel_deshuffle(self, img):
        # convert img of shape (S*H, S*W, C) to (H, W, C*S**2)
        # just test for gray sclale
        # TODO: test for RBG image
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

    def __getitem__1(self, idx):
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
        high_res = self.subpixel_deshuffle(high_res)
        
        # transform to tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)

        return low_res, high_res
    
    def __getitem__(self, idx):
        high_res = self.loader(self.hs_paths[idx])
        low_res = self.loader(self.ls_paths[idx])
        
        low_res = low_res[:, :, np.newaxis]
        high_res = high_res[:, :, np.newaxis]

        #high_res = mod_crop(high_res, 3)
        #low_res = mod_crop(low_res, 3)

        high_res = self.subpixel_deshuffle(high_res)

        # transform np image to torch tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)


        low_res = low_res - 0.5
        high_res = high_res - 0.5
        
        return low_res, high_res








        
        
        



        
        
    
        
