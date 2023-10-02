import os
import cv2
import sys
import tifffile
import torch
import random
import numpy as np
from PIL import Image
import os
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
from collections import defaultdict
from skimage import io
from torchvision import transforms as tfs
import random
from connectomics.data.dataset.augmentation_BBBC import Flip, Elastic, Grayscale, Rotate, Rescale


class ToLogits(object):
    def __init__(self, expand_dim=None):
        self.expand_dim = expand_dim

    def __call__(self, pic):
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int32, copy=True))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if self.expand_dim is not None:
            return img.unsqueeze(self.expand_dim)
        return img

class BBBC(Dataset):
    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

    def augs_mix(self, data):
        if random.random() > 0.5:
            data = self.aug_flip(data)
        if random.random() > 0.5:
            data = self.aug_rotation(data)
        # if random.random() > 0.5:
        #     data = self.aug_rescale(data)
        if random.random() > 0.5:
            data = self.aug_elastic(data)
        if random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data

    def __init__(self, dir, mode, size):
        self.size = tuple(size)  # img size after crop
        self.dir = dir
        self.mode = mode
        if (self.mode != "train") and (self.mode != "validation") and (self.mode != "test"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")

        self.flip = True
        self.crop = True
        self.data_folder = self.dir
        self.padding = 30
        self.separate_weight = True

        self.dir_img = os.path.join(self.data_folder, 'images')
        # self.dir_lb = os.path.join(self.data_folder, 'masks')
        self.dir_lb = os.path.join(self.data_folder, 'label_instance')
        self.dir_meta = os.path.join(self.data_folder, 'metadata')

        # augmentation
        self.if_scale_aug = True
        self.if_filp_aug = True
        self.if_elastic_aug = True
        self.if_intensity_aug = True
        self.if_rotation_aug = True

        if self.mode == "train":
            f_txt = open(os.path.join(self.dir_meta, 'training.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        elif self.mode == "validation":
            f_txt = open(os.path.join(self.dir_meta, 'validation.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        elif self.mode == "test":
            f_txt = open(os.path.join(self.dir_meta, 'test.txt'), 'r')
            self.id_img = [x[:-5] for x in f_txt.readlines()]  # remove .png and \n
            f_txt.close()
        else:
            raise NotImplementedError
        print('The number of %s image is %d' % (self.mode, len(self.id_img)))

        # padding for random rotation
        self.crop_size = [512, 512] # [512, 512]
        self.crop_from_origin = [0, 0]
        # self.padding = 30
        self.crop_from_origin[0] = self.crop_size[0] + 2 * self.padding
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.img_size = [520+2*self.padding, 696+2*self.padding]
        # augmentation initoalization
        self.augs_init()

    def __len__(self):
        return len(self.id_img)

    def __getitem__(self, id):

        if self.mode == 'train':
            k = random.randint(0, len(self.id_img) - 1)
            # read raw image
            imgs = tifffile.imread(os.path.join(self.dir_img, self.id_img[id] + '.tif'))
            # normalize to [0, 1]
            imgs = imgs.astype(np.float32)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
            # read label (the label is converted to instances)
            label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[id] + '.png')))

            # raw images padding
            imgs = np.pad(imgs, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
            label = np.pad(label, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')

            random_x = random.randint(0, self.img_size[0] - self.crop_from_origin[0])
            random_y = random.randint(0, self.img_size[1] - self.crop_from_origin[1])
            imgs = imgs[random_x:random_x + self.crop_from_origin[0], \
                   random_y:random_y + self.crop_from_origin[1]]
            label = label[random_x:random_x + self.crop_from_origin[0], \
                    random_y:random_y + self.crop_from_origin[1]]

            data = {'image': imgs, 'label': label}
            # print('imgs',imgs.shape) # (316, 316)
            # print('label',label.shape) # (316, 316)
            if np.random.rand() < 0.8:
                data = self.augs_mix(data)
            imgs = data['image']
            label = data['label']
            imgs = center_crop_2d(imgs, det_shape=self.crop_size)
            label = center_crop_2d(label, det_shape=self.crop_size)
            
            imgs = imgs[np.newaxis, :, :]
            imgs = np.repeat(imgs, 3, 0)  #input channels is 3
            
            imgs = torch.from_numpy(imgs)
            label = label.astype(np.float32)
            fg = label > 0
            fg = fg.astype(np.uint8)
            fg = torch.from_numpy(fg[np.newaxis, :, :].copy())

            label = torch.from_numpy(label[np.newaxis, :, :])
            pos = None
            weightmap = torch.zeros_like(label)
            return pos, imgs, label, weightmap


        elif self.mode == 'validation':
            imgs = tifffile.imread(os.path.join(self.dir_img, self.id_img[id] + '.tif'))
            # normalize to [0, 1]
            imgs = imgs.astype(np.float32)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
            # read label (the label is converted to instances)
            label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[id] + '.png')))

            imgs = np.pad(imgs, ((92, 92), (4, 4)), mode='constant')  # [704, 704]
            label = np.pad(label, ((92, 92), (4, 4)), mode='constant')

            imgs = imgs[np.newaxis, :, :]
            imgs = np.repeat(imgs, 3, 0)
            imgs = torch.from_numpy(imgs)

            fg = label > 0
            fg = fg.astype(np.uint8)
            fg = torch.from_numpy(fg[np.newaxis, :, :].copy())
            label = torch.from_numpy(label[np.newaxis, :, :].astype(np.float32))
            
            pos=None
            weightmap = torch.zeros_like(label)
            return pos, data, label, weightmap

        else:
            imgs = tifffile.imread(os.path.join(self.dir_img, self.id_img[id] + '.tif'))
            # normalize to [0, 1]
            imgs = imgs.astype(np.float32)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
            # read label (the label is converted to instances)
            label = np.asarray(Image.open(os.path.join(self.dir_lb, self.id_img[id] + '.png')))

            imgs = imgs[np.newaxis, :, :]
            imgs = np.repeat(imgs, 3, 0)
            imgs = torch.from_numpy(imgs)

            label = torch.from_numpy(label[np.newaxis, :, :].astype(np.float32))
            # fg = label > 0
            # fg = fg.astype(np.uint8)

            pos=None
            
            return label, imgs


def center_crop_2d(image, det_shape=[256, 256]):
    # To prevent overflow
    image = np.pad(image, ((10,10),(10,10)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2
    shift1 = (src_shape[1] - det_shape[1]) // 2
    assert shift0 > 0 or shift1 > 0, "overflow in center-crop"
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1]]
    return image