# Lite-AFCSNet

**Official repository for the paper "Lite-AFCSNet: An Ultra-Lightweight Asymmetric Freuency-Aware Cross-Stage Fusion Network For Acoustic Scence Classification**

## Quik Link

- **DCASE 2025 Challenge**:[https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information](https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information)

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

**Step 1:** Set root, UV4 and COM address

```powershell
$Root = "C:\Users\ROG\Desktop\DCASE_H743_Deployment"
$UV4  = "E:\Keil Core\UV4\UV4.exe"
$COM  = "COM5"
```

**Step 2:** Select the version to be burned

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
## **Data Splits**

A detailed description of the data splits (**train, test, and evaluation sets**) is available on the [official task description page](https://dcase.community/challenge2025/).

### **Training Data Restrictions**
This year's training set is limited to a **subset** of the full **development-train split** from the [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset (TAU)](https://zenodo.org/records/6337421). Specifically:
- The subset aligns with the **25% split from DCASE 2024 Task 1**.
- **Training on any other part of the TAU dataset (beyond the allowed 25% subset) is strictly prohibited** and will result in **disqualification** from the challenge. The **development-test** split can only be used to **evaluate** the system's performance.

## Baseline Complexity

The Baseline system (full fine-tune strategy) has a complexity of 61,148 parameters and 29.42 million MACs. The table below lists how the parameters
and MACs are distributed across the different layers in the network.

**According to the challenge rules the following complexity limits apply**:
* max memory for model parameters: 128 kB (Kilobyte)
* max number of MACs for inference of a 1-second audio snippet: 30 million MACs

**Model parameters may vary across different devices. As a result, the complexity limits apply individually to each device-specific model, rather than the total sum of all models.**

Model parameters of the baseline must be converted to 16-bit precision before inference of the test/evaluation set to stick to the complexity limits (61,148 * 16 bits = 61,148 * 2 B = 122,296 B <= 128 kB).

In previous years of the challenge, top-ranked teams used a technique called **quantization** that converts model paramters to 8-bit precision. In this case,
the maximum number of allowed parameters would be 128,000.


| **Description**       | **Layer**                        | **Input Shape** | **Params** | **MACs**  |
|-----------------------|----------------------------------|-----------------|------------|-----------|
| in_c[0]               | Conv2dNormActivation             | [1, 1, 256, 65] | 88         | 304,144   |
| in_c[1]               | Conv2dNormActivation             | [1, 8, 128, 33] | 2,368      | 2,506,816 |
| stages[0].b1.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 17] | 2,176      | 2,228,352 |
| stages[0].b1.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 17] | 704        | 626,816   |
| stages[0].b1.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 64, 17] | 2,112      | 2,228,288 |
| stages[0].b2.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 17] | 2,176      | 2,228,352 |
| stages[0].b2.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 17] | 704        | 626,816   |
| stages[0].b2.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 64, 17] | 2,112      | 2,228,288 |
| stages[0].b3.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 17] | 2,176      | 2,228,352 |
| stages[0].b3.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 17] | 704        | 331,904   |
| stages[0].b3.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 64, 9]  | 2,112      | 1,179,712 |
| stages[1].b4.block[0] | Conv2dNormActivation (pointwise) | [1, 32, 64, 9]  | 2,176      | 1,179,776 |
| stages[1].b4.block[1] | Conv2dNormActivation (depthwise) | [1, 64, 64, 9]  | 704        | 166,016   |
| stages[1].b4.block[2] | Conv2dNormActivation (pointwise) | [1, 64, 32, 9]  | 3,696      | 1,032,304 |
| stages[1].b5.block[0] | Conv2dNormActivation (pointwise) | [1, 56, 32, 9]  | 6,960      | 1,935,600 |
| stages[1].b5.block[1] | Conv2dNormActivation (depthwise) | [1, 120, 32, 9] | 1,320      | 311,280   |
| stages[1].b5.block[2] | Conv2dNormActivation (pointwise) | [1, 120, 32, 9] | 6,832      | 1,935,472 |
| stages[2].b6.block[0] | Conv2dNormActivation (pointwise) | [1, 56, 32, 9]  | 6,960      | 1,935,600 |
| stages[2].b6.block[1] | Conv2dNormActivation (depthwise) | [1, 120, 32, 9] | 1,320      | 311,280   |
| stages[2].b6.block[2] | Conv2dNormActivation (pointwise) | [1, 120, 32, 9] | 12,688     | 3,594,448 |
| ff_list[0]            | Conv2d                           | [1, 104, 32, 9] | 1,040      | 299,520   |
| ff_list[1]            | BatchNorm2d                      | [1, 10, 32, 9]  | 20         | 20        |
| ff_list[2]            | AdaptiveAvgPool2d                | [1, 10, 32, 9]  | -          | -         |
| **Sum**               | -                                | -               | **61,148**     | **29,419,156** |

To give an example on how MACs and parameters are calculated, let's look in detail into the module **stages[0].b3.block[1]**.
It consists of a conv2d, a batch norm, and a ReLU activation function. 

**Parameters**: The conv2d Parameters are calculated as *input_channels * output_channels * kernel_size * kernel_size*, resulting in 
1 * 64 * 3 * 3 = 576 parametes. Note that *input_channels=1* since it is a depth-wise convolution with 64 groups. The batch norm adds 64 * 2 = 128 parameters
on top, resulting in a total of 704 parameters for this *Conv2dNormActivation* module.

**MACs**: The MACs of the conv2d are calculated as *input_channels * output_channels * kernel_size * kernel_size * output_frequency_bands * output_time_frames*, resulting in 1 * 64 * 3 * 3 * 64 * 9 = 331,776 MACs.   
Note that *input_channels=1* since it is a depth-wise convolution with 64 groups. The batch norm adds 128 MACs
on top, resulting in a total of 331,904 MACs for this *Conv2dNormActivation* module.

## Baseline Results

The primary evaluation metric for the DCASE 2024 challenge Task 1 is **Macro Average Accuracy** (class-wise averaged accuracy).

The two tables below list the Macro Average Accuracy, class-wise accuracies and device-wise accuracies for the general model (Step 1), and the fine-tuned device-specific models (Step 2). The results are averaged over 4 runs.
You should obtain similar results when running the baseline system. 



### Class-wise results

| **Model**              | **Airport** | **Bus** | **Metro** | **Metro Station** | **Park** | **Public Square** | **Shopping Mall** | **Street Pedestrian** | **Street Traffic** | **Tram** | **Macro Avg. Accuracy** |
|------------------------|------------:|--------:|----------:|------------------:|---------:|------------------:|------------------:|----------------------:|-------------------:|---------:|:-----------------------:|
| General Model          |       38.94 |   62.28 |     40.60 |             50.72 |    72.03 |             29.20 |             56.04 |                 34.76 |             73.21  |    49.42 |      50.72 ± 0.47       |
| Device-specific Models |       44.43 |   64.81 |     43.87 |             48.22 |    72.75 |             32.04 |             53.14 |                 34.43 |             74.10  |    51.08 |    **51.89 ± 0.05**     |


### Device-wise results

| **Model**              | **A** | **B** | **C**     | **S1**    | **S2** | **S3**    | **S4**    | **S5** | **S6**    | **Macro Avg. Accuracy** |
|------------------------|:-----:|:-----:|:---------:|:---------:|:------:|:---------:|:---------:|:------:|:---------:|:-----------------------:|
| General Model          | 62.80 | 52.87 |  54.23    |  48.52    | 47.29  |  52.86    |  48.14    | 47.23  |  42.60    |      50.72 ± 0.47       |
| Device-specific Models | 63.98 | 55.85 |  59.09    |  48.68    | 48.74  |  52.72    |  48.14    | 47.23  |  42.60    |    **51.89 ± 0.05**     |

Note that results for devices (`s4`, `s5`, `s6`) are the same for *General Model* and *Device-specific Models*, as the *General Model* is used for unknown devices (devices that are not in the training set).

## Obtain Evaluation Set Predictions

The evaluation set will be released on **June 1st**. Detailed instructions on generating predictions for the evaluation set will be provided alongside its release.

## Inference Code

For this year's challenge, participants must submit inference code for their models. Inference code must be provided in the form of a simple installable python package. The inference package for the baseline system will be soon provided in a separate repository.


