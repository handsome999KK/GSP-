# GSP++ 
## ✨ Highlights

- **Training-free & plug-and-play**: runs directly without any pre-training or fine-tuning.
- **Unified 3D evaluation**: supports point-cloud **OOD detection** and **classification** in a single framework.
- **Lightweight deployment**: can run on a **single NVIDIA RTX 3090** GPU.



## 🛠️ Installation

### 1) Clone
```bash
git clone https://github.com/handsome999KK/GSP-.git
cd GSP-
```

### 2) Enviroment
Our experimental environment is follow [ULIP2](https://github.com/salesforce/ULIP) to establish, the code is tested with python = 3.8, CUDA==11.3 and pytorch==1.10.1. You can install GSP following (noted that you need to download the requirements.txt from [ULIP2](https://github.com/salesforce/ULIP) and put it into the folder):
```bash
conda create -n GSP++
conda activate GSP++
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### 3) Pretrained models and Dataset
We use the pretrained 3D encoder and text encoder by ULIP2, you can from [ULIP2](https://huggingface.co/datasets/SFXX/ulip/tree/main) or [here](https://drive.google.com/drive/folders/1xEblkFTEIdV1IyIlQLi792-lXfoxVCYO?usp=sharing) to download the pre-trained models, the models folder should have the following structure:
```bash

🗂️pretrained_models
|-- pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt
|-- pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt
...
🗂️data
🗂️models
....

```

For 3d ood detection task, we follow [3DOS](https://github.com/antoalli/3D_OS) to constrcut dataset, you can from [here](https://drive.google.com/drive/folders/1xEblkFTEIdV1IyIlQLi792-lXfoxVCYO?usp=sharing) to download zhe ood dataset and classification datatse, the dataset folder should have the following structure:
```bash
GSP-/data
🗂️modelnet40_normal_resampled
|-- modelnet40_test_8192pts_fps.dat
|-- modelnet40_shape_names.txt
|-- ...
🗂️scanobjectnn15_normal_resampled
|-- scanobjectnn15_test_2048pts_fps_fullshot.dat
|-- scanobjectnn15_test_2048pts_fps.dat
|-- scanobjectnn15_shape_names.txt
|-- ...
🗂️shapenetcore54_normal_resampled
|-- shapenetcore54_test_4096pts_fps_fullshot.dat
|-- shapenetcore54_test_4096pts_fps.dat
|-- shapenetcore54_shape_names.txt
|-- ....
....
```
## 📊 Evaluation
### 1) 3D ood detection
- For zero-shot on ScanObjectNN:
```bash
cd GSP-
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SR1    # eval on SR1
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SR2    # eval on SR2
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SR3    # eval on SR3
```

- For full-shot on ScanObjectNN:
```bash
cd GSP-
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SR1 --fullshot   # eval on SR1
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SR2 --fullshot   # eval on SR2
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SR3 --fullshot   # eval on SR3
```

- For zero-shot on Shapenetcore:
```bash
cd GSP-
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SN1    # eval on SN1
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SN2    # eval on SN2
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SN3    # eval on SN3
```

- For full-shot on Shapenetcore:
```bash
cd GSP-
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SN1 --fullshot   # eval on SN1
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SN2 --fullshot   # eval on SN2
python main_OOD.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --dataset_name ScanObjectNN15 --dataset_split SN3 --fullshot   # eval on SN3
```










