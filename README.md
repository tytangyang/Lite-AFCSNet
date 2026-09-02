# Lite-AFCSNet

**Official repository for the paper "Lite-AFCSNet: An Ultra-Lightweight Asymmetric Freuency-Aware Cross-Stage Fusion Network For Acoustic Scence Classification**

## Quik Link

- **DCASE 2025 Challenge**:[DCASE 2025 Task1](https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information)

## Overview

We propose Lite-AFCSNet, an ultra-lightweight asymmetric frequency-aware cross-stage fusion network for efficient acoustic scene classification. On DCASE 2025 Task 1, Lite-AFCSNet achieves **56.81% macro-average accuracy**
with only **17.23K parameters and 9.67 MMACs**, outperforming the baseline by 6.09 percentage points. Deployed on an STM32H743 MCU, it processes neural network latency in **329.1 ms (RTF = 0.402)**, enabling real-time on-device inference.

<div align="center">
  <img src="Figure/pipeline.jpg" width="50%">
  <p><i>The pipeline of the proposed method.</i></p>
</div>

<div align="center">
  <img src="Figure/architecture.jpg" width="100%">
  <p><i>Architecture of Lite-AFCSNet. The upper part shows the overall network, while the lower part illustrates basic blocks.</i></p>
</div>

<div align="center">
  <img src="Figure/Cross-Stage Feature Fusion.jpg" width="30%">
  <p><i>The proposed pooling-aligned cross-stage fusion module with LC-AFF.</i></p>
</div>

## Repository Structure
This repository is divided into three parts: the first is model training, the second is DCASE official submission and inference code, and the third is model deployment.
```
Lite-AFCSNet/
├── Train                        ← Model Training Folder
├── DCASE_T1_submission          ← DCASE official submission and packaging Folder
└── DCASE_H743_Deployment        ← Model Deployment Folder
```
### Train
```
Train/
├── dataset/                        
│   └── dcase25                   ← DCASE 2025 Task 1 dataset loading and preprocessing
├── helpers/
│   ├── init.py                   ← Random seed initialization for DataLoader workers
│   ├── complexity.py             ← Model parameter size and MACs calculation
│   └── utils.py                  ← Feature-level data augmentation utilities
├──models/
│   ├── helpers/
│   │   └── utils.py              ← Common model utility functions
│   ├── fusion.py                 ← Lightweight fusion, attention, and AsymDW modules
│   ├── multi_device_model.py     ← Device-specific model container
│   └── net.py                    ← Lite-AFCSNet network architecture
├──outputs/
│   ├── device_after_kd3/
│   │   └── multi_device_after_kd.ckpt  ← Trained checkpoint after KD and device-specific fine-tuning
├──teacher_models/
│   ├── helpers/
│   │   └── utils.py              ← Utility functions for teacher models
│   ├── fusion.py                 ← Fusion modules used by teacher models
│   ├── multi_device_model.py     ← Multi-device teacher model wrapper
│   ├── net.py                    ← Teacher model architecture
│   ├── sys4_multi_device_model.py ← Multi-device wrapper for the teacher
│   ├── sys4_net.py               ← teacher network architecture
│   └── sys4_teacher.py           ← teacher model definition and loading
├──get.py                         ← Getting the splist.csv file in the dataset folder
├── specaugment.py                ← SpecAugment implementation
├── train_base.py                 ← Base Lite-AFCSNet training
├── train_base_kd_sys4.py         ← Knowledge distillation using the teacher
├── train_device_specific.py      ← Device-specific fine-tuning
├── train_device_specific_after_kd.py ← Device-specific fine-tuning after knowledge distillation
└── verify_sys4_teacher.py         ← Verification and evaluation of the teacher
```
### DCASE_T1_submission
```
DCASE_T1_submission/
├── DCASE_T1/                        ← Main inference package
│   ├── ckpts/                       ← Model checkpoint files
│   │   ├── init.py                  ← Package initialization
│   │   └── Lite_AFCSNet.ckpt        ← Trained Lite-AFCSNet checkpoint
│   ├── models/                      ← Model architecture implementation
│   │   ├── init.py                  ← Package initialization
│   │   ├── fusion.py                ← Feature fusion and attention modules
│   │   ├── multi_device_moedl.py    ← Device-specific model container
│   │   └── net.py                   ← Lite-AFCSNet network definition
│   ├── resources/                   ← Resources required for evaluation
│   │   ├── init.py                  ← Package initialization
│   │   ├── dummy.wav                ← Dummy audio for complexity testing
│   │   └── test.xls                 ← Development-set test split
│   ├── init.py                      ← Package initialization
│   └── DCASE_T1_1.py                ← Main inference API 
├── predictions/                     ← Generated evaluation results
│   │   ├── complexity.json          ← MACs and parameter statistics
│   │   ├── model_state_dict.pt      ← Exported model parameters
│   │   ├── output.xls               ← Scene prediction results
│   │   └── test_accuacy.json        ← Development-set accuracy
├── complexity.py                    ← Model complexity calculation utilities
├── evaluate_submission.py           ← Test and evaluation inference script
├── setup.py                         ← Package installation configuration
└── test_complexity                  ← Model complexity verification script
```
### DACSE_H743_Deployment
```
DCASE_H743_Deployment/
├── firmware/                        ← On-board firmware (Keil MDK projects)
│   ├── deploy_h743_64MHz/           ← ① 64 MHz stable version (HSI)
│   └── deploy_h743_480MHz/          ← ② 480 MHz high-performance version (HSE + PLL)
├── pc_tools/                        ← PC-side Python tools and test samples
│   ├── dcase_batch_test.py          ← Batch accuracy test for 10 acoustic scenes
│   ├── dcase_fetch.py               ← Download official audio samples from Zenodo
│   ├── pc_onnx_test.py              ← PC-side ONNX inference test
│   ├── dump_logmel.py               ← Capture intermediate DSP log-Mel features
│   ├── check_board_logmel.py        ← Compare board and PC log-Mel features
│   ├── compare_logits.py            ← Compare board and PC prediction outputs
│   ├── reset_and_test.py            ← Reset board via pyOCD and run a sample test
│   ├── send_pcm.py                  ← Send a single audio sample to the board
│   ├── check_pcm.py                 ← Check audio preprocessing
│   ├── check_filterbank.py          ← Check the Mel filter bank
│   └── dataset/samples/*.wav        ← Test samples for 10 acoustic scenes
└── model/
    └── asc_base_p0_65_int8_qdq.onnx ← INT8 quantized ONNX model (QDQ format)
```


## Getting Started

1. Clone this repository.
2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n d25_t1 python=3.13
conda activate d25_t1
```

3. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

```
# for example:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for the most recent versions:
pip3 install torch torchvision torchaudio
```

4. Install requirements:

```
pip3 install -r requirements.txt
```

5. Download and extract the [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset](https://zenodo.org/records/6337421).

You should end up with a directory that contains, among other files, the following:
* A directory *audio* containing 230,350 audio files in *wav* format
* A file *meta.csv* that contains 230,350 rows with columns *filename*, *scene label*, *identifier* and *source label*

6. Specify the location of the dataset directory in the variable *dataset_dir* in file [dataset/dcase25.py](dataset/dcase25.py).
7. If you have not used [Weights and Biases](https://wandb.ai/site) for logging before, you can create a free account. On your
machine, run ```wandb login``` and copy your API key from [this](https://wandb.ai/authorize) link to the command line.

## Project Process

This repository is divided into three parts: the first is model training, the second is DCASE official submission and inference code, and the third is model deployment.

### Model Training

**Step 1:** Train a **general model** after knowledge distillation on the full training set to maximize cross-device generalization.

```
python train_base_kd_sys4.py \
  --teacher_ckpt ./sys4.ckpt
```

**Step 2:** Load the pre-trained model from **Step 1** and fine-tune for all devices in the training set (`a`, `b`, `c`, `s1`, `s2`, `s3`):

```
python train_device_specific_after_kd.py \
  --base_ckpt=<wandb_id_from_Step_1>
```

**How to specify the checkpoint?** Simply pass the Weights & Biases experiment ID (wandb_id_from_Step_1). You can find it in the Weights & Biases online dashboard.

**Hint:** When inspecting **curves** in Weights & Biases, make sure to set the **x-axis to `trainer.global_step`** instead of the default `step`. This ensures that metrics are correctly aligned across different device-specific fine-tuning phases. The default `step` counter is **shared across all phases** because a **single Weights & Biases logger instance is reused**, causing offsets in the plots.

### DCASE official submission and inference code

**Step 1:** final checkpoint from the last device-specific fine-tuning run to: 

```
DACES_T1/ckpts/Lite_AFCSNet.ckpt
```

**Step 2:** Complexity check:

```
python test_complexity.py \
  --submission_name DACES_T1 \
  --submission_index 1
```

**Step 3:** Development and evaluation inference:

```
python evaluate_submission.py \
  --submission_name DACES_T1 \
  --submission_index 1 \
  --dev_set_dir /path/to/dataset25 \
  --eval_set_dir /path/to/evaluationset
```

###  Model Deployment
Using **NUCLEO-H743ZI2** or any STM32H743 board, which requires an on-board/external 25MHz crystal oscillator for the 480MHz version.
**Step 1:** Set root, UV4 and COM address

```powershell
$Root = "C:\Users\ROG\Desktop\DCASE_H743_Deployment"
$UV4  = "E:\Keil Core\UV4\UV4.exe"
$COM  = "COM5"
```

**Step 2:** Select the version to be burned
Recommend using 480MHz
```powershell
$Proj = "$Root\firmware\deploy_h743_64MHz\Projects\NUCLEO-H743ZI\Examples\UART\UART_TwoBoards_ComPolling\MDK-ARM\Project.uvprojx"
$Proj = "$Root\firmware\deploy_h743_480MHz\Projects\NUCLEO-H743ZI\Examples\UART\UART_TwoBoards_ComPolling\MDK-ARM\Project.uvprojx"
```

**Step 3:** Compile
```powershell
& $UV4 -b $Proj -j0 -o "$Root\build.log"
```

**Step 4:** Flash
```powershell
& $UV4 -f $Proj -o "$Root\flash.log"
```

**Step 5:** Reset
```powershell
pyocd reset -t stm32h743xx
```

**Step 5:** Predict 10 types of samples
```powershell
python "$Root\pc_tools\dcase_batch_test.py" $COM
```

## Baseline Complexity

Lite-AFCSNet has a complexity of 17,231 parameters and 9,670,740 MACs. The table below lists how the parameters and MACs are distributed across the different layers in the network.

| **Description**     | **Layer**                    | **Input Shape** | **Params** |      **MACs** |
| ------------------- | ---------------------------- | --------------: | ---------: | ------------: |
| in_c[0]             | ConvBNAct, 3×3               | [1, 1, 256, 65] |         88 |       304,144 |
| in_c[1]             | ConvBNAct, 3×3               | [1, 8, 128, 33] |      2,368 |     2,506,816 |
| stage1.b1.block[0]  | Grouped PW Conv              | [1, 32, 64, 17] |        640 |       557,184 |
| stage1.b1.block[1]  | AsymDW 3×1 + 1×3             | [1, 64, 64, 17] |        512 |       417,920 |
| stage1.b1.block[2]  | Grouped PW Conv              | [1, 64, 64, 17] |        576 |       557,120 |
| stage1.b1.fuse_mode | LC-AFF                       | [1, 32, 64, 17] |          3 |            96 |
| stage1.b2.block[0]  | Grouped PW Conv              | [1, 32, 64, 17] |        640 |       557,184 |
| stage1.b2.block[1]  | AsymDW 3×1 + 1×3             | [1, 64, 64, 17] |        512 |       417,920 |
| stage1.b2.block[2]  | Grouped PW Conv              | [1, 64, 64, 17] |        576 |       557,120 |
| stage1.b2.fuse_mode | LC-AFF                       | [1, 32, 64, 17] |          3 |            96 |
| stage1.b3.block[0]  | Grouped PW Conv              | [1, 32, 64, 17] |        640 |       557,184 |
| stage1.b3.block[1]  | AsymDW 3×1 + 1×3, stride 1×2 | [1, 64, 64, 17] |        512 |       319,616 |
| stage1.b3.block[2]  | Grouped PW Conv              |  [1, 64, 64, 9] |        576 |       294,976 |
| stage1.b3.fuse_mode | LC-AFF                       |  [1, 32, 64, 9] |          3 |            96 |
| stage2.b4.block[0]  | Grouped PW Conv              |  [1, 32, 64, 9] |        640 |       295,040 |
| stage2.b4.block[1]  | AsymDW 5×1 + 1×3, stride 2×1 |  [1, 64, 64, 9] |        640 |       147,584 |
| stage2.b4.block[2]  | Grouped PW Conv              |  [1, 64, 32, 9] |      1,008 |       258,160 |
| stage2.b4.fuse_mode | LC-AFF                       |               - |          3 |             - |
| stage2.b5.block[0]  | Grouped PW Conv              |  [1, 56, 32, 9] |      1,920 |       484,080 |
| stage2.b5.block[1]  | AsymDW 5×1 + 1×3             | [1, 120, 32, 9] |      1,200 |       276,720 |
| stage2.b5.block[2]  | Grouped PW Conv              | [1, 120, 32, 9] |      1,792 |       483,952 |
| stage2.b5.fuse_mode | LC-AFF                       |  [1, 56, 32, 9] |          3 |           168 |
| stage_pool_alpha    | Learnable Max/Avg Weight     |          scalar |          1 |             - |
| channel_adjust      | 1×1 Conv                     |  [1, 32, 32, 9] |      1,792 |       516,096 |
| aff_fusion          | Cross-stage LC-AFF           |  [1, 56, 32, 9] |          3 |           168 |
| detection_head[0]   | 1×1 Conv                     |  [1, 56, 32, 9] |        560 |       161,280 |
| detection_head[1]   | BatchNorm2d                  |  [1, 10, 32, 9] |         20 |            20 |
| detection_head[2]   | AdaptiveAvgPool2d            |  [1, 10, 32, 9] |          - |             - |
| **Sum**             | -                            |               - | **17,231** | **9,670,740** |

## Results

The primary evaluation metric for the DCASE 2025 challenge Task 1 is **Macro Average Accuracy** . There are two results table below, DCASE 2025 Task 1 and On-device deployment, respectively.

### Results on DCASE 2025 Task 1

| Method | Acc. (%) | Params (K) | MACs (M) |
|---|---:|---:|---:|
| Baseline  | 50.72 | 61.15 | 29.42 |
| **MALACH25** | **61.47** | 61.15 | 29.42 |
| SNTLNTU | 59.94 | 116.34 | 10.90 |
| DynaCP | 59.58 | 61.65 | 28.94 |
| CD | 56.63 | 61.15 | 29.42 |
| SRIB | 56.06 | 61.16 | 27.86 |
| **Lite-AFCSNet** | 56.81 | **17.23** | **9.67** |

### On-device Deployment Results

| Metric | Value |
|---|---:|
| Flash | 138.26 KB |
| Peak RAM | 345.39 KB |
| DSP Latency | 72.8 ms |
| NN Latency | 329.1 ms |
| E2E Latency | 401.9 ms |
| RTF | 0.402 |
