import json
import copy
import os

fold1_dense_path = "/media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/dense_annotations_30fps/psiava_v2_dense_fold1.json"
fold2_dense_path = "/media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/dense_annotations_30fps/psiava_v2_dense_fold2.json"
train_dense_path = "/media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/dense_annotations_30fps/psiava_v2_dense_train.json"
test_dense_path = "/media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/dense_annotations_30fps/psiava_v2_dense_test.json"

fold1_dense = json.load(open(fold1_dense_path))
fold2_dense = json.load(open(fold2_dense_path))
train_dense = json.load(open(train_dense_path))
test_dense = json.load(open(test_dense_path))

annotated_images = {}

for dico in train_dense['images']:
    file_name = dico["file_name"]
    case, image = file_name.split("/")
    if case in annotated_images.keys():
        annotated_images[case].append(image)
    else:
        annotated_images[case] = [image]

unlabeled_train_dense = copy.deepcopy(train_dense)
unlabeled_train_dense["annotations"] = []
unlabeled_train_dense["images"] = []

dense_keyframe_path = "/media/SSD1/nayobi/All_datasets/PSI-AVA/dense_keyframes/"

case_list = os.listdir(dense_keyframe_path)
case_list.sort()

id = 1
keys = list(annotated_images.keys())
keys.sort()
num_neighbors=1
neighbor_list=[]
for case in keys:
    annotated_images[case].sort()
    for frame in annotated_images[case]:
        #all_images.remove(image_name)
        num_frame = int(frame.split('.')[0])
        
        for i in range(1, num_neighbors+1):
            
            ant_neigh = num_frame-i
            penant_neigh = num_frame-i-1
            ant_neigh_name = '0'*(9-len(str(ant_neigh))) + str(ant_neigh) + '.jpg'
            penant_neigh_name = '0'*(9-len(str(penant_neigh))) + str(penant_neigh) + '.jpg'
            
            fol_neigh = num_frame + i
            aftfol_neigh = num_frame + i + 1
            fol_neigh_name = '0'*(9-len(str(fol_neigh))) + str(fol_neigh) + '.jpg'
            aftfol_neigh_name = '0'*(9-len(str(aftfol_neigh))) + str(aftfol_neigh) + '.jpg'

            if not os.path.isfile(os.path.join(dense_keyframe_path, case, ant_neigh_name)):
                if os.path.isfile(os.path.join(dense_keyframe_path, case, penant_neigh_name)) and penant_neigh_name not in neighbor_list:
                    neighbor_list.append(case + '/' + penant_neigh_name)
                if os.path.isfile(os.path.join(dense_keyframe_path, case, fol_neigh_name)) and fol_neigh_name not in neighbor_list:
                    neighbor_list.append(case + '/' + fol_neigh_name)
            
            elif not os.path.isfile(os.path.join(dense_keyframe_path, case, fol_neigh_name)):
                if os.path.isfile(os.path.join(dense_keyframe_path, case, aftfol_neigh_name)) and aftfol_neigh_name not in neighbor_list:
                    neighbor_list.append(case + '/' + aftfol_neigh_name)
                if os.path.isfile(os.path.join(dense_keyframe_path, case, ant_neigh_name)) and ant_neigh_name not in neighbor_list:    
                    neighbor_list.append(case + '/' + ant_neigh_name)
            
            else:
                if ant_neigh_name not in neighbor_list:  
                    neighbor_list.append(case + '/' + ant_neigh_name)
                if fol_neigh_name not in neighbor_list:
                    neighbor_list.append(case + '/' + fol_neigh_name)
    neighbor_list.sort()
    for image in neighbor_list:
        dico={}
        dico["id"] = id
        dico["file_name"] = image
        dico["width"] = 1280
        dico["height"] = 800
        dico["license"] = 1
        dico["coco_url"] = ""
        dico["flickr_url"] = ""
        id +=1
        unlabeled_train_dense["images"].append(dico)
    neighbor_list=[]

with open('/home/eugenie/EndoVis/data/GraSP/unlabeled_splits/train/unlabeled_neighbors_train_dense.json', 'w') as outfile:
    json.dump(unlabeled_train_dense, outfile)























