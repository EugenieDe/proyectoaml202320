CUDA_VISIBLE_DEVICES=3 python train_net.py --num-gpus 1 \
--config-file configs/endovis2018/Res50.yaml \
OUTPUT_DIR output/endovis2018/resnet50_base_test0 \
MODEL.WEIGHTS https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl