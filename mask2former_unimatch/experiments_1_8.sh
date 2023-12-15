CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 \
--config-file configs/endovis2018/1_8/SwinS.yaml \
OUTPUT_DIR output/endovis2018/1_8/SwinS/new_transforms \
 MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl"

CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 \
--config-file configs/endovis2018/1_8/Res50.yaml \
OUTPUT_DIR output/endovis2018/1_8/resnet50/new_transforms \
MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl

CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 \
--config-file configs/endovis2018/1_8/SwinT.yaml \
OUTPUT_DIR output/endovis2018/1_8/SwinT/new_transforms \
MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"

CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 \
--config-file configs/endovis2018/1_8/Res101.yaml \
OUTPUT_DIR output/endovis2018/1_8/resnet101/new_transforms \
MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R101_bs16_50ep/model_final_eba159.pkl"



