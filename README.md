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
conda create -n GSP
conda activate GSP
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### 3) Pretrained models and Dataset
WWe use the pretrained 3D encoder and text encoder by ULIP2, you can from [here]([https://github.com/salesforce/ULIP](https://huggingface.co/datasets/SFXX/ulip/tree/main))

