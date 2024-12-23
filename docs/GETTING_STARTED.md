# Environment Setup for RadarDistill

This guide outlines the steps to set up the development environment for RadarDistill. The tested environment includes:

- **CUDA**: 11.3
- **Python**: 3.7+
- **PyTorch**: 1.10.0
- **CUDA** 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)

## **Installation**

### **1. Create and Activate a Conda Environment**
```bash
conda create -n RadarDistill python=3.7 -y
conda activate RadarDistill
apt update && apt install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    ninja-build \
    libglib2.0-0 \
    libxrender-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
```

### **2. Install PyTorch and CUDA Toolkit**
```bash
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
```

### **3. Install `RadarDistll`**
```bash
git clone https://github.com/your-username/RadarDistill.git
cd RadarDistill
pip install -r requirements.txt
python setup.py develop
pip install spconv-cu113 torch-scatter==2.1.1
cd pcdet/ops/basicblock
sh make.sh
```


## NuScenes Dataset Preparation for Distillation

This guide explains how to prepare the NuScenes dataset for use with the distillation framework. It assumes you have already cloned the repository and installed the necessary dependencies.

### **1. Download NuScenes Dataset**
Download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the dataset as follows:
```plaintext
RadarDistill
├── data
│   ├── nuscenes
│   │   ├── v1.0-trainval
│   │   │   ├── samples
│   │   │   ├── sweeps
│   │   │   ├── maps
│   │   │   ├── v1.0-trainval
│   │   ├── v1.0-test
├── pcdet
├── tools
```
---

### **2. Dataset Preparation Commands**
To preprocess the NuScenes dataset for distillation, run the following command:
```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset_distill --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_distill.yaml \
    --version v1.0-trainval
```

### **3. Generated Files**
After running the commands, the following files will be created:
```plaintext
RadarDistill
├── data
│   ├── nuscenes
│   │   ├── gt_database_10sweeps_with_radar_withvelo
│   │   ├── nuscenes_infos_6radar_10sweeps_train.pkl
│   │   ├── nuscenes_infos_6radar_10sweeps_val.pkl
│   │   ├── nuscenes_dbinfos_10sweeps_with_radar_withvelo.pkl
```
## Pretrained Models
If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the pretrained [DeepLabV3 model](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and place within the `checkpoints` directory. Please make sure the [kornia](https://github.com/kornia/kornia) is installed since it is needed for `CaDDN`.
```
OpenPCDet
├── checkpoints
│   ├── deeplabv3_resnet101_coco-586e9e4e.pth
├── data
├── pcdet
├── tools
```

## Training & Testing


### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```


### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```
