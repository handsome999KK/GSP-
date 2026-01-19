#SR OOD
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ScanObjectNN15 --dataset_split SR3

python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt --dataset_name ScanObjectNN15 --dataset_split SR3

python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt --dataset_name ScanObjectNN15 --dataset_split SR3

python main_OOD.py --model ULIP_PN_MLP --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/ULIP-main/scripts/checkpoint_pointmlp.pt --dataset_name ScanObjectNN15 --dataset_split SR3

python main_OOD.py --model ULIP_PN_SSG --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/checkpoint_pointnet2_ssg.pt --dataset_name ScanObjectNN15 --dataset_split SR3

#SN OOD
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt --dataset_name ShapeNetCore54 --dataset_split SN3

python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ShapeNetCore54 --dataset_split SN3
#SN cls
python main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt --dataset_name ShapeNetCore54

python main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr  /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ShapeNetCore54

#SR cls
python main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ScanObjectNN15

python main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt --dataset_name ScanObjectNN15

python main_cls.py --model ULIP_PN_MLP --evaluate_3d --npoints 2048 --test_ckpt_addr /home/chentiankai987/Code/GSP/ULIP-main/scripts/checkpoint_pointmlp.pt --dataset_name ScanObjectNN15

#MN cls
python main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 8192 --test_ckpt_addr /home/chentiankai987/Code/GSP/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ModelNet40

python main_cls.py --model ULIP_PN_SSG --evaluate_3d --npoints 8192 --test_ckpt_addr /home/chentiankai987/Code/GSP/checkpoint_pointnet2_ssg.pt --dataset_name ModelNet40





# ####################### Docker

export CUDA_VISIBLE_DEVICES=4
# SR OOD
python3 main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr /workspace/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ScanObjectNN15 --dataset_split SR3

# SR cls
python3 main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr /workspace/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ScanObjectNN15

# SN cls
python3 main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /workspace/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ShapeNetCore54

python3 main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /workspace/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt --dataset_name ShapeNetCore54

# MN cls
python3 main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 8192 --test_ckpt_addr /workspace/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt --dataset_name ModelNet40

python3 main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /workspace/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt --dataset_name ModelNet40

python3 main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /workspace/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt --dataset_name ModelNet40

python3 main_cls.py --model ULIP_PointBERT --evaluate_3d --npoints 4096 --test_ckpt_addr /workspace/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt --dataset_name ModelNet40