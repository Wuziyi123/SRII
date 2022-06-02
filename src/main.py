import math
import os

from cifar100.iCaRL import iCaRLmodel
from miniImageNet.iCaRL_image import miniImagemodel
from src.backbone.vgg import vgg
from backbone.ResNet import resnet18_cbam, resnet50_cbam
from backbone.ResNet_down import branch_resnet18_cbam, branch_resnet50_cbam
import torch
# from utils.my_utils import *
import numpy as np
import random


def main(parser_data):
    os.chdir(r'../')
    isCifar = True
    feature_extractor = None
    branch_feature_extractor = None

    # Adjust the resnet architecture according to the dataset
    if parser_data.benchmark == 'cifar100':
        isCifar = True
    elif parser_data.benchmark == 'miniImageNet':
        isCifar = False
    else:
        print("we don't have this dataset!")
        exit(-1)

    network = parser_data.network

    if network == 'resnet':
        feature_extractor = resnet18_cbam(isCifar=isCifar)
        branch_feature_extractor = branch_resnet18_cbam(isCifar=isCifar)
    elif network == 'vgg':
        feature_extractor = vgg()
        branch_feature_extractor = vgg()
    else:
        print("we don't adapt this network to the backbone!")
        exit(-1)

    numclass = 10
    img_size = 32
    task_size = 10
    memory_size = 2000

    epochs = parser_data.epochs
    batch_size = parser_data.batch_size
    learning_rate = parser_data.learning_rate
    data_path = parser_data.input_data

    if isCifar:
        model = iCaRLmodel(data_path, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
        branch_model = iCaRLmodel(data_path, numclass,branch_feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
    else:
        model = miniImagemodel(data_path, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
        branch_model = miniImagemodel(data_path, numclass, branch_feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)

    accuracy = 0
    branch_model.model.fc = None

    # If the pretrained saved weight file address is specified:
    print(parser_data.pre_trained)
    if parser_data.pre_trained != "":
        if os.path.isfile(parser_data.pre_trained):
            print("=> loading checkpoint '{}'".format(parser_data.pre_trained))
            checkpoint = torch.load(parser_data.pre_trained)
            model.model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(parser_data.pre_trained))

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
            exit(-1)
            model.afterTrain(accuracy, branch=True, down_model=down_model)
        else:
            model.afterTrain(accuracy, branch=False)


if __name__ == "__main__":
    version = torch.version.__version__[:5]  # example: 1.6.0
    # Because the official mixed accuracy training is only supported after 1.6.0,
    # it must be greater than or equal to 1.6.0
    if version < "1.6.0":
        raise EnvironmentError("pytorch version must be 1.6.0 or above")

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # The type of training equipment
    parser.add_argument('--device', default='cuda:0', help='device')
    # The type of backbone
    parser.add_argument('--network', default='resnet', type=str, help='the type of backbone',
                        choices=['resnet', 'vgg'], metavar='BACKBONE (RESNET OR VGG?)')
    # learning rate
    parser.add_argument('--learning_rate', default=2.0, type=float, help='learning rate')
    # training epochs
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    # Root directory of training dataset
    parser.add_argument('--input_data', default='dataset', type=str, help='path of dataset',
                        choices=['dataset', 'mini-imagenet/images'], metavar='PATH_TO_DATA')
    # Root directory of training dataset
    parser.add_argument('--benchmark', default='cifar100', type=str, help='benchmark dataset',
                        choices=['cifar100', 'miniImageNet'], metavar='BENCHMARK_DATASET')
    # Storage address
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # pretrained weights  pretrained/cifar100-pretrained.pth.tar
    parser.add_argument('--pre_trained', default='pretrained/cifar100-pretrained.pth.tar', type=str, help='pre-trained model', metavar='PRE-TRAINED MODEL')
    # batch size
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    print(args)

    # Check whether the folder for saving weights exists. If it does not exist, create ...
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
