import json
import os 
import pycocotools.mask as m
import numpy as np
import copy
import pandas as pd

pred_path = "/home/eugenie/EndoVis/proyectoaml202320/mask2former_unimatch/output/endovis2018/inference/train/1_8/SwinS/new_transforms/inference/coco_instances_results.json"
test_path = "/home/eugenie/EndoVis/data/endovis2018/RobotSeg2018_inst_class_train.json"

pred = json.load(open(pred_path))
test = json.load(open(test_path))

pred_full = copy.deepcopy(test)


pred_full['annotations'] = []
anns = pred 
id = 1
image_id = 1
score_max=0

for ann in anns:
    if ann['image_id'] == image_id:
        if ann['score']>=score_max:
            score_max= ann['score']
            max_ann = ann
        if ann['score']>0.9:
            # now only support compressed RLE format as segmentation results
            segm = copy.deepcopy(ann['segmentation'])
            segm['counts'] = segm['counts'].encode('utf-8')
            ann['area'] = int(m.area(segm))
            del ann['bbox']
            if not 'bbox' in ann:
                ann['bbox'] = (m.toBbox(segm)).astype(int).tolist()
            ann['id'] = int(id)
            ann['iscrowd'] = 0
            pred_full['annotations'].append(ann)
            id+=1
    else:
        if pred_full['annotations'][-1]['image_id']!= image_id:
            if max_ann['score']>0.5:
                segm = copy.deepcopy(max_ann['segmentation'])
                segm['counts'] = segm['counts'].encode('utf-8')
                max_ann['area'] = int(m.area(segm))
                del max_ann['bbox']
                if not 'bbox' in max_ann:
                    max_ann['bbox'] = (m.toBbox(segm)).astype(int).tolist()
                max_ann['id'] = int(id)
                max_ann['iscrowd'] = 0
                pred_full['annotations'].append(max_ann)
                id+=1
            else:
                max_ann = {}
                max_ann['segmentation'] = {}
                max_ann['image_id'] = image_id
                max_ann['area'] = 0
                max_ann['bbox'] = [0.0, 0.0, 0.0, 0.0]
                max_ann['id'] = int(id)
                max_ann['iscrowd'] = 0
                max_ann['score'] = 1
                max_ann['category_id'] = 8
                pred_full['annotations'].append(max_ann)
                id+=1
            """    
            try:
                segm = copy.deepcopy(max_ann['segmentation'])
            except:
                breakpoint()
            segm['counts'] = segm['counts'].encode('utf-8')
            max_ann['area'] = int(m.area(segm))
            del max_ann['bbox']
            if not 'bbox' in max_ann:
                max_ann['bbox'] = (m.toBbox(segm)).astype(int).tolist()
            max_ann['id'] = int(id)
            max_ann['iscrowd'] = 0
            pred_full['annotations'].append(max_ann)
            id+=1
            """
        image_id += 1
        score_max=0
        max_ann={}
        if ann['score']>score_max:
            score_max= ann['score']
            max_ann = ann
        if ann['score']>0.9:
            # now only support compressed RLE format as segmentation results
            segm = copy.deepcopy(ann['segmentation'])
            segm['counts'] = segm['counts'].encode('utf-8')
            ann['area'] = int(m.area(segm))
            del ann['bbox']
            if not 'bbox' in ann:
                ann['bbox'] = (m.toBbox(segm)).astype(int).tolist()
            ann['id'] = int(id)
            ann['iscrowd'] = 0
            pred_full['annotations'].append(ann)
            id+=1



anns = copy.deepcopy(pred_full['annotations'])
ims = copy.deepcopy(pred_full['images'])
dico = {}
for ann in anns:
    image_id = ann['image_id']
    if image_id in dico.keys():
        dico[image_id].append(ann)
    else:
        dico[image_id] = []
        dico[image_id].append(ann)

for k in dico.keys():
    for ann in dico[k]:
        ann["ToDelete"] = False
#breakpoint()
for k in dico.keys():
    if len(dico[k]) != 1:
        for i in range(0, len(dico[k])-1):
            segm = dico[k][i]['segmentation']
            mask = m.decode(segm)
            for j in range(i+1, len(dico[k])):
                segm2 = dico[k][j]['segmentation']
                mask2 = m.decode(segm2)
                iou = np.sum(np.logical_and(mask, mask2))/(np.sum(np.logical_or(mask, mask2)))
                if iou>0.9:
                    if dico[k][i]['score']>dico[k][j]['score']:
                        dico[k][j]["ToDelete"] = True                 
                    else: 
                        dico[k][i]["ToDelete"] = True
        #breakpoint()

for k in dico.keys():
    for ann in dico[k]:
        #if k==520:
        #    breakpoint()
        if ann["ToDelete"] == True:
            dico[k].remove(ann)
#breakpoint()

pred_full['annotations']=[]
for image_id in dico:
    for ann in dico[image_id]:
        try:
            del ann['ToDelete']
        except:
            breakpoint()
        pred_full['annotations'].append(ann)

#breakpoint()

for ann in pred_full['annotations']:
    bbox = ann['bbox']
    if bbox != [0.0, 0.0, 0.0, 0.0]:
        bbox[0] = bbox[0]/ann['segmentation']['size'][1]
        bbox[2] = bbox[2]/ann['segmentation']['size'][0]
        bbox[1] = bbox[1]/ann['segmentation']['size'][1]
        bbox[3] = bbox[3]/ann['segmentation']['size'][0]

seq = pred_full['images'][0]['file_name'].split("_")[0] + "_" + pred_full['images'][0]['file_name'].split("_")[1]
im_id = 0
csv_dict = []
id = 0
count_id = 0

for im in pred_full['images']:
    curr_seq = im['file_name'].split("_")[0] + "_" + im['file_name'].split("_")[1]
    dict_ann = {}
    dict_ann['seq'] = curr_seq
    if curr_seq == seq:
        dict_ann['im_id'] = im_id
    else:
        im_id=0
        dict_ann['im_id'] = im_id
    im_id +=1
    count_id +=1
    seq = curr_seq

    k=0
    for ann in pred_full['annotations']:
        if ann['image_id'] == count_id:
           #breakpoint()
           k=1
           dict_ann['id'] = id
           id +=1
           bbox = ann['bbox']
           dict_ann['xmin'] = bbox[0]
           dict_ann['ymin'] = bbox[1]
           dict_ann['xmax'] = bbox[2]
           dict_ann['ymax'] = bbox[3]
           dict_ann['category'] = ann['category_id']
           dict_ann['score'] = ann['score']
           csv_dict.append(dict_ann)
           dict_ann = copy.deepcopy(dict_ann)
        else:
            if k==1:
                k=2
        if k==2:
            break
#breakpoint()            


filename = "/home/eugenie/EndoVis/proyectoaml202320/mask2former_unimatch/output/endovis2018/inference/train/1_8/SwinS/new_transforms/inference/val_pred.csv"
dataframe = pd.DataFrame(csv_dict)
#breakpoint()
dataframe.to_csv(filename, index=False, header=False)


 


#json.dump(pred_full, open("/home/eugenie/These/proyectoaml202320/mask2former_unimatch/output/endovis2018/inference/1_2/SwinS/new_transforms/inference/full_instances_results.json",'w'))










