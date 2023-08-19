# Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/Wuziyi123/SRII/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

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
  Download `miniImageNet` and make it looks like:
  ```shell
  mini-imagenet/
  ├── images
	  ├── n0210891500001298.jpg  
	  ├── n0287152500001298.jpg 
	  ...
  ├── test.csv
  ├── val.csv
  ├── train.csv
  └── imagenet_class_index.json
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
sh ./main.sh --input_data dataset --pre_trained pretrained/cifar100-pretrained.pth.tar --network resnet
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



# Acknowledgements

Special thanks to https://github.com/DRSAD/iCaRL for his iCaRL Networks
 implementation of which parts were used for this implementation. More
  details of iCaRL at https://arxiv.org/abs/1611.07725

***