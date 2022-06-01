# Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/Wuziyi123/SRII/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

***


# Paper

- ***An Incremental Learning Apporoach for Sustainable Regional Isolation and Integration*** 

This repository is the official implementation of [SRII: An Incremental Learning Apporoach for Sustainable Regional Isolation and Integration](https://*******). 

***


# Incremental-Learning-Benchmark
Evaluate class incremental learning tasks shifting with popular continual learning algorithms.

The benchmarks come from the following contributions:

- A-GEM: [paper](https://openreview.net/forum?id=Hkf2_sC5FX) (Efficient lifelong learning with A-GEM)
- EMR: [paper](https://arxiv.org/abs/1902.10486) (On Tiny Episodic Memories in Continual Learning)
- iCaRL: [paper](https://ieeexplore.ieee.org/document/8100070) (icarl: Incremental classifier and representation learning)
- LUCIR: [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html) (Learning a Unified Classifier Incrementally via Rebalancing)
- LwF: [paper](https://ieeexplore.ieee.org/document/8107520) (Learning without forgetting)
- EWC: [paper](https://arxiv.org/abs/1612.00796) (Overcoming catastrophic forgetting in neural networks)
- ABD: [paper](https://ieeexplore.ieee.org/document/9711051) (Always be dreaming: A new approach for data-free class-incremental learning)
- SCR: [paper](https://ieeexplore.ieee.org/document/9522763) (Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning)
- S&B: [paper](https://arxiv.org/abs/2107.01349) (Split-and-Bridge: Adaptable Class Incremental Learning within a Single Neural Network)
- E2E: [paper](https://arxiv.org/abs/1807.09536) (End-to-end incremental learning)
- Bic: [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.html) (Large scale incremental learning)

***


# Pre-trained

## Installation

### Requirements
To install requirements:

```requirements
python == 3.6
pytorch == 1.8.1
torch == 1.7.0
torchvision >= 0.8
numpy == 1.19.5
matplotlib == 3.3.4
opencv-python == 4.5.1.48
```

### Setup

 * Install anaconda: https://www.anaconda.com/distribution/
 * set up conda environmet & python 3.6, ex: `conda create --name <env_name> python=3.6`

```setup
conda activate <env_name>
conda env create -f environment.yml -p <anaconda>/envs/<env_name>
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## DataSet
We conduct experiments on commonly used incremental learning bencnmarks: CIFAR100, miniImageNet.
  1. `CIFAR100` is available at [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html). Download `CIFAR100` and put it under the `dataset directory`.
  2. `miniImageNet` is available at [Our Google Drive](https://drive.google.com/file/d/15WB2Q5vawJxai9vHrw5FGbPBKAeTTfBY).
  * Download `miniImageNet` and make it looks like:
  ```shell
  mini-imagenet/
  ├── images
	  ├── n0210891500001298.jpg  
	  ├── n0287152500001298.jpg 
	  ...
  ├── test.csv
  ├── val.csv
  └── train.csv
  ```


## Models

You can download `pretrained model` here:

- [My pre-trained model](https://drive.google.com/file/d/1iSd466hB69USclAyxuqK07LxzHFe8SCA/view?usp=sharing) trained on CIFAR100. 

You can download `test model` here, then put it under the `model directory`:

- [My test model](https://drive.google.com/file/d/1pJDbDGa2uUCfCABCTfrh-c_CejzZ9C21/view?usp=sharing) trained on CIFAR100.

***


# Training

All commands should be run under the project root directory.

To train the model(s) in the paper, run this command:

```train
sh ./main.sh --input_data dataset --pre_trained pretrained/cifar100-pretrained.pth.tar
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

***


# Evaluation

To evaluate my model, run:

```eval
sh ./test.sh --model_file model/test-20classes.pth.tar
```


>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

***


# Results

In the following tables, we provide detailed test accuracies of each method under different settings of benchmark datasets and CNN models. The reported results are the mean accuracies averaged over 5 or 10 runs.

**Upper Bound**. Joint training, equivalent to retraining the model on all known data, works best. Therefore, it is often considered a "performance upper bound for incremental learning", but the training cost is too high.

**10-Stage or 10 Stages**. We divide the dataset containing 100 classes into 10 incremental steps and add 10 classes in each incremental step.

**5-Stage or 5 Stages**. We divide the dataset containing 100 classes into 5 incremental steps and take the learning of 5 classes as the starting point. The first incremental step learns 15 classes, and the next four incremental steps add 20 classes at a time.

These results are also reported in the figures of the paper.

![image](https://github.com/Wuziyi123/SRII/blob/main/figures/Figure_5.png)

## [(a) CIFAR-100 10-Stage (with 2000 image coreset)](https://paperswithcode.com/sota/class-incremental-learning-on-cifar100)

tasks | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | Avg |
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
A-GEM | 85.0 | 58.42 | 48.3 | 44.39 | 43.7 | 41.4 | 40.48 | 37.26 | 31.74 | 26.91 | 45.76  
EMR | 82 | 61.5 | 54.67 | 50.74 | 47.83 | 44.42 | 42.71 | 36.73 | 34.17 | 31.78 | 48.66 
iCaRL | 84.9 | 73.7 | 69.17 | 64.75 | 61.94 | 60.17 | 58.3 | 54.99 | 53.6 | 50.83 | 63.24  
LUCIR | 89.1 | 72.2 | 63.43 | 56.17 | 53 | 49.87 | 49.3 | 46.31 | 43.81 | 42.09 | 56.53 
LwF | 85.8 | 58.8 | 53.62 | 48.52 | 42.02 | 38.24 | 35.86 | 33.16 | 29.43 | 25.74 | 45.12 
EWC | 86.1 | 66.1 | 60.57 | 53.75 | 47.42 | 43.88 | 41.07 | 39.24 | 35.83 | 31.33 | 50.53 
ABD | **91.5** | 74.2 | 70.2 | 57.8 | 52.98 | 46 | 43.36 | 38.59 | 36.52 | 33.2 | 54.44 
SCR | 86 | 76.7 | 74.1 | 68.7 | 65.5 | 63.9 | 60.03 | 58.9 | 54.91 | 51.08 | 65.98 
S&B | 87.2 | 81.47 | 77.52 | 73.64 | 69.15 | 64.66 | 61.55 | 59.05 | 55.31 | 52.29 | 68.18 
Upper Bound | 87.3 | 84.72 | 82.43 | 81.59 | 79.74 | 78.64 | 78.42 | 77.11 | 76.85 | 76.32 | **80.30** 
Ours (SRII) | **90.5** | **85.5** | **82.52** | **80** | **76.08** | **72.11** | **68.74** | **64.14** | **61.97** | **60.25** | **74.18**

## (b) miniImageNet 10-Stage (with 2000 image coreset)

tasks | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | Avg |
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
S&B | 89.26 | 83.77 | 79.8 | 75.12 | 70.46 | 66.93 | 63.08 | 62.35 | 57.63 | 54.72 | 70.31  
LwF | 88.1 | 81.2 | 72.2 | 63.57 | 55.3 | 49.85 | 44.86 | 40.77 | 37.14 | 32.3 | 56.53 
Bic | 90.8 | 80.18 | 75.53 | 71.23 | 67.65 | 62.58 | 58.19 | 54.86 | 51.54 | 47.88 | 66.04  
E2E | 90.45 | 79.68 | 72.53 | 67.93 | 62.65 | 58.58 | 54.49 | 50.86 | 49.14 | 43.88 | 63.02 
ABD | 93.3 | 78.39 | 73.86 | 66.54 | 58.94 | 53.24 | 48.01 | 43.73 | 39.93 | 38.03 | 59.4 
Upper Bound | 89.5 | 86.71 | 84.13 | 83.25 | 81.56 | 82.14 | 80.57 | 79.11 | 78.88 | 78.32 | **82.42** 
Ours (SRII) | **92.3** | **87.69** | **84.18** | **81.57** | **77.57** | **73.84** | **71.04** | **67.36** | **64.19** | **62.48** | **76.22**

## (c) miniImageNet 5-Stage (with 2000 image coreset)

tasks | 5 | 20 | 40 | 60 | 80 | 100 | Avg |
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
S&B | 89.26 | 83.77 | 79.8 | 75.12 | 70.46 | 66.93 | 63.08 
LwF | 88.1 | 81.2 | 72.2 | 63.57 | 55.3 | 49.85 | 44.86 
Bic | 90.8 | 80.18 | 75.53 | 71.23 | 67.65 | 62.58 | 58.19   
E2E | 90.45 | 79.68 | 72.53 | 67.93 | 62.65 | 58.58 | 54.49  
ABD | 93.3 | 78.39 | 73.86 | 66.54 | 58.94 | 53.24 | 48.01  
Upper Bound | 89.5 | 86.71 | 84.13 | 83.25 | 81.56 | 82.14 | 80.57  
Ours (SRII) | **92.3** | **87.69** | **84.18** | **81.57** | **77.57** | **73.84** | **71.04** 

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

***


# Acknowledgements

Special thanks to https://github.com/DRSAD/iCaRL for his iCaRL Networks
 implementation of which parts were used for this implementation. More
  details of iCaRL at https://arxiv.org/abs/1611.07725

***