#CUDA_VISIBLE_DEVICES=3 python /home/eugenie/EndoVis/proyectoaml202320/mask2former_unimatch/train_net.py --num-gpus 1 \
#--config-file configs/endovis2018/1_2/SwinS.yaml \
#OUTPUT_DIR output/endovis2018/1_2/SwinS/proportionnal_loss/conf/0.90 \
#MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl"

CUDA_VISIBLE_DEVICES=0 python /home/eugenie/EndoVis/proyectoaml202320/mask2former_unimatch/train_net.py --num-gpus 1 \
--config-file configs/endovis2018/1_2/SwinS.yaml \
OUTPUT_DIR output/endovis2018/1_2/SwinS/new_transforms/batch4 \
MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl"