import math
import os

from iCaRL import iCaRLmodel
# from iCaRL_image import miniImagemodel
from ResNet import resnet18_cbam, resnet34_cbam, resnet50_cbam
from ResNet_down import branch_resnet18_cbam, branch_resnet34_cbam, branch_resnet50_cbam
import torch
from my_utils import *
import numpy as np
import random

numclass = 10
feature_extractor = resnet18_cbam()
branch_feature_extractor = branch_resnet18_cbam()
img_size = 32
batch_size = 128
task_size = 10
memory_size = 2000
epochs = 1
learning_rate = 2.0


model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
# model = miniImagemodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
branch_model = iCaRLmodel(numclass,branch_feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

accuracy = 0
branch_model.model.fc = None

for i in range(10):

    model.beforeTrain()
    if not i:
        for c in range(2):  # basic training, including initial training, regional split
            accuracy = model.train(i, c, branch=False)
    else:
        for c in range(3):  # incremental training
            if c == 1:
                branch_feature_extractor.reset_params()
                branch_model.new_beforeTrain(i, model)
                accuracy, down_model = branch_model.train(i, c, model, branch=True)
            else:
                accuracy = model.train(i, c, branch=False)
    if i and (c == 1):
        model.afterTrain(accuracy, branch=True, down_model=down_model)
    else:
        model.afterTrain(accuracy, branch=False)
