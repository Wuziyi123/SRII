>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [SRII: An Incremental Learning Apporoach for Sustainable Regional Isolation and Integration](https://*******). 


## Installation
***
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
 * set up conda environmet w/ python 3.8, ex: `conda create --name <env_name> python=3.8.5`
 * `conda activate <env_name>`

```setup
conda activate <env_name>
conda env create -f environment.yml -p <anaconda>/envs/<env_name>
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training
***
All commands should be run under the project root directory.

To train the model(s) in the paper, run this command:

```train
sh ./main.sh
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation
***
To evaluate my model, run:

```eval
sh ./test.sh --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models
***
You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results
***
Results are generated for various task sizes. See the main text for full details.
Our model achieves the following performance on :

### [CIFAR-100 10-Stage (with 2000 image coreset)](https://paperswithcode.com/sota/image-classification-on-imagenet)

tasks | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
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

### [miniImageNet 10-Stage (with 2000 image coreset)](https://paperswithcode.com/sota/image-classification-on-imagenet)

tasks | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
S&B | 89.26 | 83.77 | 79.8 | 44.39 | 43.7 | 41.4 | 40.48 | 37.26 | 31.74 | 26.91 | 45.76  
LwF | 82 | 61.5 | 54.67 | 50.74 | 47.83 | 44.42 | 42.71 | 36.73 | 34.17 | 31.78 | 48.66 
Bic | 84.9 | 73.7 | 69.17 | 64.75 | 61.94 | 60.17 | 58.3 | 54.99 | 53.6 | 50.83 | 63.24  
E2E | 89.1 | 72.2 | 63.43 | 56.17 | 53 | 49.87 | 49.3 | 46.31 | 43.81 | 42.09 | 56.53 
ABD | 85.8 | 58.8 | 53.62 | 48.52 | 42.02 | 38.24 | 35.86 | 33.16 | 29.43 | 25.74 | 45.12 
Upper Bound | 87.3 | 84.72 | 82.43 | 81.59 | 79.74 | 78.64 | 78.42 | 77.11 | 76.85 | 76.32 | **80.30** 
Ours (SRII) | **92.3** | **87.69** | **84.18** | **81.57** | **77.57** | **73.84** | **71.04** | **67.36** | **64.19** | **62.48** | **76.22**

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing
***

>📋  Pick a licence and describe how to contribute to your code repository.


## Acknowledgements
***
Special thanks to https://github.com/DRSAD/iCaRL for his iCaRL Networks
 implementation of which parts were used for this implementation. More
  details of iCaRL at https://arxiv.org/abs/1611.07725
