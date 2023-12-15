CUDA_VISIBLE_DEVICES=3 python train_net.py --num-gpus 1 \
--config-file configs/endovis2018/1_2/SwinS.yaml \
OUTPUT_DIR output/endovis2018/1_2/SwinS/other_transforms_u_no_rand_apply_less_strong/conf/0.95 \
MODEL.WEIGHTS "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl"