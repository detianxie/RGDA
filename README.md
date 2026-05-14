RGDA: Reliability-Gated Decoupled Adaptation

Official PyTorch implementation of the paper: "Reliability-Gated Decoupled Adaptation for Robust Test-Time Adaptation in Noisy Industrial Defect Detection" (The Visual Computer).

Abstract: Industrial defect detection systems often suffer severe performance degradation under extreme noise and distribution shift during deployment. Existing test-time adaptation (TTA) methods suffer from unstable updates caused by unreliable samples and coupled optimization. This work introduces a reliability-gated decoupled adaptation (RGDA) framework to separate statistical alignment from parameter optimization. A reliability gating mechanism filters high-uncertainty samples, while anchor-based regularization stabilizes updates...

🚀 Main Contributions

Decoupled Adaptation Architecture: We decouple statistical alignment and gradient optimization, fundamentally mitigating the confirmation bias under extreme noise.

Reliability Gating Mechanism: A strict entropy-based thresholding module ($\tau$) purifies the backpropagated gradients.

Anchor-based Regularization: An adaptive architectural constraint ($\lambda$) that dynamically balances plasticity and stability based on the inherent inductive bias of various models (e.g., CNNs vs. Vision Transformers).

🛠️ Environment Setup

# zenodo
https://doi.org/10.5281/zenodo.20155296
cd RGDA

# Create conda environment
conda create -n rgda python=3.9.25
conda activate rgda

# Install dependencies
pip install -r requirements.txt


📂 Data Preparation

The experiments are conducted on industrial defect datasets. Please organize your datasets in the following structure:

data/
├── RIAWELC/
│   ├── train/
│   └── test/
└── SWRD/
    ├── level1/
    ├── level2/
    └── level3/


🏃 Quick Start

1. Run RGDA (Ours)

To evaluate the proposed RGDA framework on ConvNeXt under severe noise (Level 3):

python evaluate_RGDA.py \
    --test_data_dir ./data/SWRD/level3 \
    --model_path ./convtextrun_best_model.pth \
    --model_name resnet50 \
    --tta_lr 1e-3 \
    --e_margin 0.9 \
    --fisher_alpha 0.1


2. Run Baseline Methods

We provide unified implementations for state-of-the-art TTA methods. For example, to run Tent:

python evaluate_tent.py \
    --test_data_dir ./data/SWRD/level3 \
    --model_path ./weights/resnet50.pth \
    --model_name resnet50


Other baselines (Tent,CoTTA, SAR, EATA) can be executed similarly using their respective scripts.

📧 Contact

For any questions, please feel free to open an issue or contact xiedetian@mail.shiep.edu.cn.
