import os

import torch
import torch.utils.data
from PIL import Image
import sys

from glob import glob
from os.path import *
import json
import random 
import numpy as np

from ....maskrcnn_benchmark.structures.bounding_box import BoxList


class EndoVis2018Dataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "Bipolar_Forceps",
        "Prograsp_Forceps",
        "Large_Needle_Driver",
        "__Vessel_Sealer__",
        "__Grasping_Retractor__",
        "Monopolar_Curved_Scissors",
        "Ultrasound_Probe",
        "Suction_Instrument",
        "Clip_Applier",
    )

    def __init__(self, img_dir, ann_file, split, transforms=None):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.image_set = split
        self.transforms = transforms
        
        self.img_list = glob(join(self.img_dir, '*.png'))
        self.img_list.sort()

        self.annotations = json.load(open(ann_file))

        self.anno = self.annotations['annotations']
        self.image = self.annotations['images']

        cls = EndoVis2018Dataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = find_anno(self.image, 'id', index)[0]['file_name']
        image_root = join(self.img_dir, img_id)
        img = Image.open(image_root).convert("RGB")

        if self.split == 'train_u':
            target=None
            if self.transforms[1] is not None:
                transform_base = self.transforms[0]
                transform_s = self.transforms[1]
                img = transform_base(img)
                img_s1 = transform_s(img)
                img_s2 = transform_s(img)
                cutmix1 = obtain_cutmix_box(256)
                cutmix2 = obtain_cutmix_box(256)
            return img, img_s1, img_s2, cutmix1, cutmix2, target, index

        else:
            target = self.get_groundtruth(index)
            target = target.clip_to_image(remove_empty=True)

            if self.transforms is not None:
                img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.image)

    def get_groundtruth(self, index):
        anno = find_anno(self.anno, 'image_id', index)
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []

        for obj in target:

            boxes.append(obj['bbox'])
            gt_classes.append(obj['category_id'])

        im_info = (target[0]['height'], target[0]['width'])
        

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = find_anno(self.image, 'id', index)[0]
        return {"height": img_id['height'], "width": img_id['width']}

    def map_class_id_to_class_name(self, class_id):
        return EndoVis2018Dataset.CLASSES[class_id]


def find_anno(dict, index, key):
    list_anno=[]
    for i in range(0,len(dict)):
        if dict[i][key]==index:
            list_anno.append(dict[i])
    list_anno 

def obtain_cutmix_box(img_size, prob=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > prob:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

