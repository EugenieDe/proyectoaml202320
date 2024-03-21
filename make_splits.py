import json

split = "1_4"

endo_train_json_path = "/home/eugenie/EndoVis/data/endovis2018/RobotSeg2018_inst_class_train.json" 
endo_val_json_path = "/home/eugenie/EndoVis/data/endovis2018/RobotSeg2018_inst_class_val.json"

endo_splits_labeled_path = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/sorted/" + split + "/labeled.txt"
endo_splits_unlabeled_path = "/home/eugenie/EndoVis/proyectoaml202320/unimatch/splits/endovis2018/sorted/" + split + "/unlabeled.txt"

endo_train = json.load(open(endo_train_json_path))

with open(endo_splits_labeled_path, 'r') as f:
    id_labeled = f.read().splitlines()
with open(endo_splits_unlabeled_path, 'r') as f:
    id_unlabeled = f.read().splitlines()

list_name_im_labeled = []
list_name_im_unlabeled = []

for i in range(0, len(id_labeled)):
    im_name = id_labeled[i].split("/")[-1]
    list_name_im_labeled.append(im_name)

for i in range(0, len(id_unlabeled)):
    im_name = id_unlabeled[i].split("/")[-1]
    list_name_im_unlabeled.append(im_name)


dict_endo_train_labeled = {}
dict_endo_train_labeled['info'] = endo_train['info']
dict_endo_train_labeled['licenses'] = endo_train['licenses']
dict_endo_train_labeled['categories'] = endo_train['categories']
dict_endo_train_labeled['images'] = []
dict_endo_train_labeled['annotations'] = []

dict_endo_train_unlabeled = {}
dict_endo_train_unlabeled['info'] = endo_train['info']
dict_endo_train_unlabeled['licenses'] = endo_train['licenses']
dict_endo_train_unlabeled['categories'] = endo_train['categories']
dict_endo_train_unlabeled['images'] = []
dict_endo_train_unlabeled['annotations'] = []

list_im_id_labeled=[]
list_im_id_unlabeled=[]
j=0
k=0
for i in range(0, len(endo_train['images'])):
    im_name = endo_train['images'][i]['file_name']
    if im_name in list_name_im_labeled:
        dict_to_append = endo_train['images'][i]
        dict_to_append['new_id'] = j
        dict_endo_train_labeled['images'].append(dict_to_append)
        list_im_id_labeled.append(endo_train['images'][i]['id'])
        j+=1
    else:
        dict_to_append = endo_train['images'][i]
        dict_to_append['new_id'] = k
        dict_endo_train_unlabeled['images'].append(dict_to_append)
        list_im_id_unlabeled.append(endo_train['images'][i]['id'])
        k+=1

k=0
for i in range(0, len(endo_train['annotations'])):
    im_id = endo_train['annotations'][i]['image_id']
    if im_id in list_im_id_labeled:
        dict_to_append = endo_train['annotations'][i]
        dict_to_append['new_id'] = k
        dict_endo_train_labeled['annotations'].append(dict_to_append)
        k+=1


with open('/home/eugenie/EndoVis/data/endovis2018/train/splits/' + split + '/labeled.json', 'w') as outfile:
    json.dump(dict_endo_train_labeled, outfile)
with open('/home/eugenie/EndoVis/data/endovis2018/train/splits/' + split + '/unlabeled.json', 'w') as outfile:
    json.dump(dict_endo_train_unlabeled, outfile)

