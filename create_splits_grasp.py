import os
import os.path as osp
import random

def gen_split():
    endovis2018_train_path = "/media/SSD0/ihernandez/ENDOVIS/data/endovis2018/train"
    endovis2018_im_train_path = osp.join(endovis2018_train_path, 'images')
    train_names = os.listdir(endovis2018_im_train_path)

    endovis2018_val_path = "/media/SSD0/ihernandez/ENDOVIS/data/endovis2018/val"
    endovis2018_im_val_path = osp.join(endovis2018_val_path, 'images')
    val_names = os.listdir(endovis2018_im_val_path)

    train_names.sort()
    val_names.sort() 

    train_names_complete = []
    val_names_complete = []

    for name in train_names:
        image = name
        name = osp.join(endovis2018_im_train_path, name)
        endovis2018_mask_train_path = osp.join(osp.join(endovis2018_train_path, 'annotations'), image)
        name = name + ' ' + endovis2018_mask_train_path + '\n'
        train_names_complete.append(name)

    for name in val_names:
        image = name
        name = osp.join(endovis2018_im_val_path, name)
        endovis2018_mask_val_path = osp.join(osp.join(endovis2018_val_path, 'annotations'), image)
        name = name + ' ' + endovis2018_mask_val_path + '\n'
        val_names_complete.append(name)

    train_names_complete = sorted(train_names_complete, key=lambda x: random.random())

    split_path = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted"

    split_train = open(osp.join(split_path, 'train.txt'), 'x')
    split_val = open(osp.join(split_path, 'val.txt'), 'x')

    split_train.writelines(train_names_complete)
    split_val.writelines(val_names_complete)

    split_train.close()
    split_val.close()


def split(num):   
    split_path = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/unsorted/1_" + str(num)

    endovis2018_train_path = "/media/SSD0/ihernandez/ENDOVIS/data/endovis2018/train"
    endovis2018_im_train_path = osp.join(endovis2018_train_path, 'images')
    train_names = os.listdir(endovis2018_im_train_path)

    train_names.sort()

    labeled = []
    unlabeled = []

    for i in range(0,len(train_names)):
        name = train_names[i]
        image = name
        name = osp.join(endovis2018_im_train_path, name)
        endovis2018_mask_train_path = osp.join(osp.join(endovis2018_train_path, 'annotations'), image)
        name = name + ' ' + endovis2018_mask_train_path + '\n'
        if i%num == 0:
            labeled.append(name)
        else:
            unlabeled.append(name)

    labeled = sorted(labeled, key=lambda x: random.random())
    unlabeled = sorted(unlabeled, key=lambda x: random.random())

    split_labeled = open(osp.join(split_path, 'labeled.txt'), 'x')
    split_unlabeled = open(osp.join(split_path, 'unlabeled.txt'), 'x')

    split_labeled.writelines(labeled)
    split_unlabeled.writelines(unlabeled)

    split_labeled.close()
    split_unlabeled.close()

#gen_split()
split(16)
