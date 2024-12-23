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


