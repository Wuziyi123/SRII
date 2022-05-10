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

numclass = 20
feature_extractor = resnet18_cbam()
branch_feature_extractor = branch_resnet18_cbam()
img_size = 32
batch_size = 128
task_size = 10
memory_size = 2000
epochs = 100
learning_rate = 2.0


model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
# model = miniImagemodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
branch_model = iCaRLmodel(numclass,branch_feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

accuracy = 0
branch_model.model.fc = None

for i in range(10):

    i = 1
    c = 1
    if i and c == 1:
        if os.path.isfile("temp-base.pth.tar"):
            print("=> loading checkpoint '{}'".format("temp-base.pth.tar"))
            checkpoint = torch.load("temp-base.pth.tar")
            model.cfg_mask = checkpoint['cfg_mask']
            model.exemplar_set = checkpoint['exemplar_set']
            model.class_mean_set = checkpoint['class_mean_set']
            model.old_model = checkpoint['old_model']
            model.model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format("temp-base.pth.tar"))

    model.beforeTrain()
    if not i:
        for c in range(2):
            accuracy = model.train(i, c, branch=False)
    else:
        for c in range(2):
            c = 1

            if c == 1:
                branch_feature_extractor.reset_params()
                branch_model.new_beforeTrain(i, model)
                accuracy, down_model = branch_model.train(i, c, model, branch=True)
                break
            else:
                accuracy = model.train(i, c, branch=False)
    if i and (c == 1):
        model.afterTrain(accuracy, branch=True, down_model=down_model)
        exit(-1)
    else:
        model.afterTrain(accuracy, branch=False)
