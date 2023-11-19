# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def build_transforms(cfg, mode='train_l', is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 1 #0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    if is_train:
        if mode == "train_l":
            transform = T.Compose(
            [
               T.Resize(min_size, max_size),
               T.RandomHorizontalFlip(flip_prob),
               T.RandomRotation(), 
               T.RandomVerticalFlip(),
               T.ToTensor(),
               normalize_transform,
            ]
        )
        else:
            transform_base = T.Compose(
            [
               T.Resize(min_size, max_size),
               T.RandomHorizontalFlip(flip_prob),
               T.RandomRotation(), 
               T.RandomVerticalFlip(),
               T.ToTensor(),
               normalize_transform,
            ]
            )
            transform_s = T.Compose(
            [
               T.Resize(min_size, max_size),
               T.RandomHorizontalFlip(flip_prob),
               T.RandomRotation(), 
               T.RandomVerticalFlip(),
               T.ColorJitter(),
               T.RandomGrayscale(),
               T.Blur(),
               #T.CutMix(),
               T.ToTensor(),
               normalize_transform,
            ]
        )
            return transform_base, transform_s

    else:
        transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform

"""
def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
"""