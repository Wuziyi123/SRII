import numpy as np
import torch
from numpy import ndarray
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn

from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

from ResNet import resnet18_cbam

import os
from iCaRL import iCaRLmodel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

numclass = 30
feature_extractor = resnet18_cbam()
img_size = 32
batch_size = 128
task_size = 10
memory_size = 2000
epochs = 100
learning_rate = 2.0

load = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
model = load.model

# # split-refine-refine-C
mask = []
channel_weights = []

if os.path.isfile("yanma-7.pth.tar"):
    print("=> loading checkpoint '{}'".format("yanma-7.pth.tar"))
    checkpoint = torch.load("yanma-7.pth.tar")
    mask = checkpoint['yanma']
    images = checkpoint['images']
    channel_weights = checkpoint['channel_weights']
else:
    print("=> no checkpoint found at '{}'".format("yanma-7.pth.tar"))


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# print(mask)  # List[(8,128,16,16),]
c = mask[1].data.cpu().numpy()  # 8,128,16,16
d = channel_weights[1].data.cpu().numpy()  # (128,)
d = np.expand_dims(d, axis=0)  # 1,128
# e1 = c[7, :, :, :]  # 128,16,16
# -
# img = images[1].data.cpu().numpy()  # (3,32,32)
# img = img.transpose(1, 2, 0)
# e1 = np.expand_dims(e1, axis=0)  # 1,128,16,16


# def returnCAM(feature_conv, weight_softmax, class_idx):
def returnCAM(feature_conv, channel_weights, i):
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape  # 1,128,16,16
    output_cam = []

    # channel_weights = np.ones((1, 128))
    cam = channel_weights.dot(feature_conv.reshape((nc, h * w)))
    print(cam.shape)
    cam = cam.reshape(h, w)
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))

    activate = cv2.resize(cam_img, (32, 32))
    filename = "CAM{}-1.jpg".format(i)
    cv2.imwrite(filename, activate)

    return output_cam


for i in range(10):
    e1 = c[30+i, :, :, :]  # 128,16,16
    e1 = np.expand_dims(e1, axis=0)  # 1,128,16,16


    CAMs = returnCAM(e1, d, i)

    # img = cv2.imread(img_path)
    # height, width, _ = img.shape

    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (32, 32)), cv2.COLORMAP_JET)

    # result = heatmap * 0.3 + img * 0.7
    result = heatmap
    filename = "CAM{}.jpg".format(i)
    cv2.imwrite(filename, result)
    # cv2.imwrite('CAM0.jpg', result)


import cv2
import numpy as np

import matplotlib.pyplot as plt

# for i in range(5):
#     img = images[i].data.cpu().numpy()  # (3,32,32)
#     img = img.transpose(1, 2, 0)
#     img = img[:,:,(2,1,0)]  # 32,32,3
#     r,g,b = [img[:,:,i] for i in range(3)]
#     img_gray = r*0.299+g*0.587+b*0.114
#
#     plt.imshow(img_gray, cmap="gray")
#     plt.axis('off')
#     plt.show()

for i in range(10):
    img = images[30+i].data.cpu().numpy()  # (3,32,32)
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = np.zeros(img.shape, dtype=np.float32)
    cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # cv2.imshow("norm", np.uint8(result*255.0))
    filename = "filename-{}.jpg".format(i)
    cv2.imwrite(filename, np.uint8(result*255.0))











# *************************************************************

# # plt.imshow(img)
# # plt.axis('off')
# # plt.show()
#
# img = img[:,:,(2,1,0)]  # 32,32,3
# r,g,b = [img[:,:,i] for i in range(3)]
# img_gray = r*0.299+g*0.587+b*0.114
#
# # # plt.imshow(img_gray)
# # # plt.axis('off')
# # # plt.show()
#
# plt.imshow(img_gray,cmap="gray")
# plt.axis('off')
# plt.show()
