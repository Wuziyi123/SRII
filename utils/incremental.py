import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn

import iCaRL
from ResNet import resnet18_cbam
import torch.optim as optim

import os

from ResNet_down import branch_resnet18_cbam
from iCaRL import iCaRLmodel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

numclass = 20
feature_extractor = resnet18_cbam()
img_size = 32
batch_size = 128
task_size = 10
memory_size = 2000
epochs = 100
learning_rate = 2.0
branch_feature_extractor = branch_resnet18_cbam()

load = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
branch_model = iCaRLmodel(10,branch_feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
model = load.model
downmodel = branch_model.model

# split-refine-refine-C

if os.path.isfile("temp-merge-11.pth.tar"):
    print("=> loading checkpoint '{}'".format("temp-merge-11.pth.tar"))
    checkpoint = torch.load("temp-merge-11.pth.tar")
    # load.cfg_mask = checkpoint['cfg_mask']
    # weight = checkpoint['weight']
    # bias = checkpoint['bias']
    # load.exemplar_set = checkpoint['exemplar_set']
    # load.class_mean_set = checkpoint['class_mean_set']
    model.load_state_dict(checkpoint['state_dict'])
    downmodel.load_state_dict(checkpoint['down_branch'], False)
else:
    print("=> no checkpoint found at '{}'".format("temp-merge-11.pth.tar"))

# if os.path.isfile("temp-merge-1.pth.tar"):
#     print("=> loading checkpoint '{}'".format("temp-merge-1.pth.tar"))
#     checkpoint = torch.load("temp-merge-1.pth.tar")
#     # load.cfg_mask = checkpoint['cfg_mask']
#     # weight = checkpoint['weight']
#     # bias = checkpoint['bias']
#     # load.exemplar_set = checkpoint['exemplar_set']
#     # load.class_mean_set = checkpoint['class_mean_set']
#     model.load_state_dict(checkpoint['state_dict'])
#     downmodel.load_state_dict(checkpoint['down_branch'])
# else:
#     print("=> no checkpoint found at '{}'".format("temp-merge-1.pth.tar"))

# if os.path.isfile("temp-base.pth.tar"):
#     print("=> loading checkpoint '{}'".format("temp-base.pth.tar"))
#     checkpoint = torch.load("temp-base.pth.tar")
#     load.cfg_mask = checkpoint['cfg_mask']
#     weight = checkpoint['weight']
#     bias = checkpoint['bias']
#     load.exemplar_set = checkpoint['exemplar_set']
#     load.class_mean_set = checkpoint['class_mean_set']
#     model.load_state_dict(checkpoint['state_dict'])
# else:
#     print("=> no checkpoint found at '{}'".format("temp-base.pth.tar"))

# if os.path.isfile("best.pth.tar"):
#     print("=> loading checkpoint '{}'".format("best.pth.tar"))
#     checkpoint = torch.load("best.pth.tar")
#     # load.cfg_mask = checkpoint['cfg_mask']
#     # weight = checkpoint['weight']
#     # bias = checkpoint['bias']
#     # load.exemplar_set = checkpoint['exemplar_set']
#     # load.class_mean_set = checkpoint['class_mean_set']
#     model.load_state_dict(checkpoint['state_dict'])
#     downmodel.load_state_dict(checkpoint['down_branch'], False)
# else:
#     print("=> no checkpoint found at '{}'".format("best.pth.tar"))

model.to(device)
downmodel.to(device)

# model.fc.weight.data = weight
# model.fc.bias.data = bias

cfg_mask = load.cfg_mask
layer_id_in_cfg = 0
start_mask = torch.ones(3)
# end_mask = cfg_mask[layer_id_in_cfg]

# cc = model.feature
# for m0 in cc.modules():
#     if isinstance(m0, nn.BatchNorm2d):
#         bn_mask = torch.ones_like(end_mask) - end_mask
#
#         idx0 = np.squeeze(np.argwhere(np.asarray(bn_mask.cpu().numpy())))
#         # m0.weight.data[idx0] = 0
#         # m0.bias.data[idx0] = 0
#
#         layer_id_in_cfg += 1
#         start_mask = end_mask.clone()
#         if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
#             end_mask = cfg_mask[layer_id_in_cfg]


def get_train_refine_dataloader(classes):
    load.train_dataset.getTrainData(classes, load.exemplar_set)

    train_loader = DataLoader(dataset=load.train_dataset,
                              shuffle=True,
                              batch_size=load.batchsize)

    return train_loader


def get_test_refine_dataloader(classes):
    load.test_dataset.getTestData(classes)

    test_loader = DataLoader(dataset=load.test_dataset,
                             shuffle=True,
                             batch_size=128)

    return test_loader


def get_train_and_test_dataloader(classes):
    if classes[0] == 0:
        load.exemplar_set = []
    load.train_dataset.getTrainData(classes, load.exemplar_set)

    load.test_dataset.getTestData(classes)

    train_loader = DataLoader(dataset=load.train_dataset,
                              shuffle=True,
                              batch_size=128)

    test_loader = DataLoader(dataset=load.test_dataset,
                             shuffle=True,
                             batch_size=128)

    return train_loader, test_loader


def beforeTrain():
    load.model.eval()
    load.model.Incremental_learning(30)
    load.model.train()
    load.model.to(device)


def test():  # 80.0t

    classes = [0, 20]   # 73.5-83.8  77.4-81.9-79.65  75.9-84.9-80.4  83.35-84.0-82.69  83.75-84.3n-83.5
    #  84.4-84.2-84.60  <--->   84.55-86.19o-82.9n-0.1
    #  84.55-85.4o-83.69-0.12  84.6-84.3o-84.9n   84.75-85.0o-84.5  85.05-85.69o-84.4n  85.25-85.3o-85.2n
    _, test_loader = get_train_and_test_dataloader(classes)  # 85.05-84.3o-85.8n-1.4

    # for name, m0 in model.named_modules():
    #     if isinstance(m0, nn.Linear):
    #         c = m0.weight.data.clone()
    #         d = m0.bias.data.clone()

    model.eval()
    downmodel.eval()

    correct, total = 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        downmodel.getValue = True
        model.up_branch = True

        with torch.no_grad():
            # outputs = model(imgs)  # outputs = (128,20)

            x1n, g1, x2n, g2, x3n, g3 = downmodel(imgs)
            material_from_down = [x1n, g1, x2n, g2, x3n, g3]
            _, outputs = model(imgs, material_from_down, 1)

        model.up_branch = False
        downmodel.getValue = False

            # out = model.feature_extractor(imgs)  # 128,512

            # outputs = model.fc(out)
            # c = outputs[:, 20:].clone()
            # output_new = model.fc_new(c)
            # outputs[:, 20:] = output_new


            # outputs = load.classify(imgs)

            # old_predicts = load.classify(imgs)
            # outputs = model(imgs)
            # predicts = torch.max(outputs, dim=1)[1]
            # predicts[:10] = old_predicts

        # predicts = outputs

        predicts = torch.max(outputs, dim=1)[1]
        # predicts = torch.add(predicts, 10)

        # with torch.no_grad():
        #     new_num = outputs.shape[1] - 11
        #     new_index = torch.nonzero(predicts > new_num).squeeze(1)
        #     new_imgs = imgs[new_index, ::]
        #     outputs_new = downmodel(new_imgs)
        # predicts_new = torch.max(outputs_new, dim=1)[1]
        # new_start = int(outputs.shape[1] - 10)
        # predicts_new = torch.add(predicts_new, new_start)
        # predicts[new_index] = predicts_new

        # predicts = 10 + torch.max(outputs[:, 10:], dim=1)[1]
        # predicts = torch.max(outputs[:, :10], dim=1)[1]

        # e = np.array(np.nonzero(predicts.cpu() != labels.cpu())).squeeze().tolist()
        # pe = predicts[e]
        # la = labels[e]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    print(str(accuracy.item()))


test()


# print(load.model)
def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1


def acc_test(testloader):
    load.model.eval()
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(testloader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = load.model(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    load.model.train()
    return accuracy


def train():

    # beforeTrain()

    # optimizer
    # pg = [p for p in model.parameters() if p.requires_grad]
    # pg = [p for p in model.fc.parameters()]
    # opt = optim.SGD(pg, lr=learning_rate, weight_decay=0.00001, momentum=0.9, nesterov=True)
    opt = optim.SGD(load.model.parameters(), lr=load.learning_rate, weight_decay=0, momentum=0.9, nesterov=True)

    # opt = optim.SGD(load.model.parameters(), lr=learning_rate, weight_decay=0)
    classes = [10, 20]
    train_loader = get_train_refine_dataloader(classes)
    test_loader = get_test_refine_dataloader([0, 30])

    for epoch in range(100):

        if epoch == 48:
            # pg = [p for p in model.parameters() if p.requires_grad]
            for p in opt.param_groups:
                p['lr'] = learning_rate / 2.5
        elif epoch == 62:
            for p in opt.param_groups:
                p['lr'] = learning_rate / 12.5
        elif epoch == 80:
            for p in opt.param_groups:
                p['lr'] = learning_rate / 62.5

        for step, (indexs, images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            # new_index = torch.nonzero(target > 9).squeeze(1)
            enable_amp = True if "cuda" in device.type else False
            scaler = amp.GradScaler(enabled=enable_amp)

            if step or epoch:
                cfg_mask = load.cfg_mask
                layer_id_in_cfg = 0
                end_mask = cfg_mask[layer_id_in_cfg]
                for m0 in model.modules():
                    if isinstance(m0, nn.BatchNorm2d):
                        bn_mask = torch.ones_like(end_mask) - end_mask

                        m0.weight.grad.data.mul_(bn_mask)
                        m0.bias.grad.data.mul_(bn_mask)

                        layer_id_in_cfg += 1
                        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                            end_mask = cfg_mask[layer_id_in_cfg]


            with amp.autocast(enabled=enable_amp):
                output = load.model(images)  # output = (128,10)
                target = iCaRL.get_one_hot(target, 30)
                output, target = output.to(device), target.to(device)
                loss_value = F.binary_cross_entropy_with_logits(output, target)

            opt.zero_grad()
            scaler.scale(loss_value).backward()  # backward

            # updateBN(load.model.feature)

            # num = 20
            # load.model.fc.weight.grad.data[:num].mul_(0)
            # load.model.fc.bias.grad.data[:num].mul_(0)

            cfg_mask = load.cfg_mask
            layer_id_in_cfg = 0
            # skip = [7, 8, 12, 13, 17, 18]
            start_mask = torch.ones(3)
            end_mask = cfg_mask[layer_id_in_cfg]

            for name, m0 in model.named_modules():
                if isinstance(m0, nn.BatchNorm2d):
                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()

                    if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                        end_mask = cfg_mask[layer_id_in_cfg]
                elif isinstance(m0, nn.Conv2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    m0.weight.grad.data[:, idx0, :, :].mul_(0)
                    m0.weight.grad.data[idx1, :, :, :].mul_(0)


            scaler.step(opt)  # optimize
            scaler.update()

            if step and step % 30 == 0:
                print('epoch:%d,step:%d,loss:%.6f' % (epoch, step, loss_value.item()))

            end = len(train_loader) - 1
            if epoch == 99 and step == end:
                print("save!")
                torch.save({
                    'state_dict': model.state_dict(),
                    'cfg_mask': load.cfg_mask,
                    'optimizer': opt.state_dict(),
                    'class_mean_set': load.class_mean_set,
                    'exemplar_set': load.exemplar_set,
                    'weight': load.weight,
                    'bias': load.bias,
                }, "incremental-1.pth.tar")
        accuracy = acc_test(test_loader)
        print('epoch:%d,accuracy:%.3f,' % (epoch, accuracy))


# train()

