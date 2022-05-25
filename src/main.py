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


def main(parser_data):
    numclass = 10
    feature_extractor = resnet18_cbam()
    branch_feature_extractor = branch_resnet18_cbam()
    img_size = 32
    task_size = 10
    memory_size = 2000
    epochs = 100
    batch_size = parser_data.batch_size
    learning_rate = parser_data.learning_rate

    os.chdir(r'../')

    model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
    # model = miniImagemodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
    branch_model = iCaRLmodel(numclass,branch_feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)

    accuracy = 0
    branch_model.model.fc = None

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
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
    # 因为使用的官方的混合精度训练是1.6.0后才支持的，所以必须大于等于1.6.0
    if version < "1.6.0":
        raise EnvironmentError("pytorch version must be 1.6.0 or above")

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # learning rate
    parser.add_argument('--learning_rate', default=2.0, type=float, help='learning rate')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='./', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    # ./save_weights/resNetFpn-model-4.pth
    parser.add_argument('--pre_trained', default='', type=str, help='resume from checkpoint')
    # 训练的batch size
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
