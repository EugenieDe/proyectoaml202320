# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            if target==None:
                image= t(image)
            else:
                image, target = t(image, target)
        if target==None:
            return image
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size)
                )

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target == None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target == None:
                return image
            target = target.transpose(0)
        return image, target
    
class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            if target == None:
                return image
            target = F.vflip(target)
        return image, target

class RandomRotation(object):
    def __init__(self, angle=90, prob=0.5):
        self.angle = angle
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            r = torchvision.transforms.RandomRotation.get_params((-self.angle, self.angle))
            image = F.rotate(image,r)
            if target == None:
                return image
            target = F.rotate(target,r)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        if target == None:
            return F.to_tensor(image)
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target == None:
                return image
        return image, target

class ColorJitter(object):
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
        return image

class RandomGrayscale(object):
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, image):
        image = torchvision.transforms.RandomGrayscale(p=self.prob)
        return image
    
class Blur(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        

    def __call__(self, image):
        if random.random() < self.prob:
            sigma = np.random.uniform(0.1, 2.0)
            image = F.gaussian_blur(image, kernel_size=7, sigma=sigma)
        return image
    
class CutMix(object):
    def __init__(self, img_size=256, prob=0.5):
        self.prob = prob
        self.img_size = img_size
    
    def obtain_cutmix_box(self, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
        mask = torch.zeros(self.img_size, self.img_size)
        if random.random() > self.prob:
            return mask

        size = np.random.uniform(size_min, size_max) * self.img_size * self.img_size
        while True:
            ratio = np.random.uniform(ratio_1, ratio_2)
            cutmix_w = int(np.sqrt(size / ratio))
            cutmix_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, self.img_size)
            y = np.random.randint(0, self.img_size)

            if x + cutmix_w <= self.img_size and y + cutmix_h <= self.img_size:
                break

        mask[y:y + cutmix_h, x:x + cutmix_w] = 1

        return mask

    def __call__(self):
        cutmixbox = self.obtain_cutmix_box()
        return cutmixbox