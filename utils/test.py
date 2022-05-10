import numpy as np
import torch
from sklearn import preprocessing
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn

import iCaRL
from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
from ResNet_down import branch_resnet18_cbam
import torch.optim as optim

import sys
import os
from iCaRL import iCaRLmodel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

numclass = 20
feature_extractor = resnet18_cbam()
branch_feature_extractor = branch_resnet18_cbam()
img_size = 32
batch_size = 128
task_size = 10
memory_size = 2000
epochs = 100
learning_rate = 2.0

load = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate)
model = load.model

# split-refine-refine-C

# if os.path.isfile("refine-c.pth.tar"):
#     print("=> loading checkpoint '{}'".format("refine-c.pth.tar"))
#     checkpoint = torch.load("refine-c.pth.tar")
#     cfg_mask = checkpoint['cfg_mask']
#     weight = checkpoint['weight']
#     bias = checkpoint['bias']
#     load.exemplar_set = checkpoint['exemplar_set']
#     load.class_mean_set = checkpoint['class_mean_set']
#     model.load_state_dict(checkpoint['state_dict'])
# else:
#     print("=> no checkpoint found at '{}'".format("refine-c.pth.tar"))

# if os.path.isfile("refine.pth.tar"):
#     print("=> loading checkpoint '{}'".format("refine.pth.tar"))
#     checkpoint = torch.load("refine.pth.tar")
#     cfg_mask = checkpoint['cfg_mask']
#     weight = checkpoint['weight']
#     bias = checkpoint['bias']
#     load.exemplar_set = checkpoint['exemplar_set']
#     load.class_mean_set = checkpoint['class_mean_set']
#     model.load_state_dict(checkpoint['state_dict'])
# else:
#     print("=> no checkpoint found at '{}'".format("refine.pth.tar"))

# if os.path.isfile("split-iter.pth.tar"):
#     print("=> loading checkpoint '{}'".format("split-iter.pth.tar"))
#     checkpoint = torch.load("split-iter.pth.tar")
#     load.cfg_mask = checkpoint['cfg_mask']
#     weight = checkpoint['weight']
#     bias = checkpoint['bias']
#     load.exemplar_set = checkpoint['exemplar_set']
#     load.class_mean_set = checkpoint['class_mean_set']
#     model.load_state_dict(checkpoint['state_dict'])
# else:
#     print("=> no checkpoint found at '{}'".format("split-iter.pth.tar"))

if os.path.isfile("temp.pth.tar"):
    print("=> loading checkpoint '{}'".format("temp.pth.tar"))
    checkpoint = torch.load("temp.pth.tar")
    cfg_mask = checkpoint['cfg_mask']
    weight = checkpoint['weight']
    bias = checkpoint['bias']
    load.exemplar_set = checkpoint['exemplar_set']
    load.class_mean_set = checkpoint['class_mean_set']
    try:
        checkpoint["model"] = {k: v for k, v in checkpoint["state_dict"].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(checkpoint["model"], strict=False)
    except KeyError as e:
        print("")
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    print("=> no checkpoint found at '{}'".format("temp.pth.tar"))

model.to(device)

# model.fc.weight.data = weight
# model.fc.bias.data = bias
# cfg_mask = load.cfg_mask
# layer_id_in_cfg = 0
# start_mask = torch.ones(3)
# end_mask = cfg_mask[layer_id_in_cfg]

# cc = model.feature
# for m0 in cc.modules():
#     if isinstance(m0, nn.BatchNorm2d):
#         bn_mask = torch.ones_like(end_mask) - end_mask
#
#         # m0.weight.data = 0
#         # m0.bias.data = 0
#
#         # idx0 = np.squeeze(np.argwhere(np.asarray(bn_mask.cpu().numpy())))
#         # m0.weight.data[idx0] = 0
#         # m0.bias.data[idx0] = 0
#
#         # index = end_mask.long()
#
#         # m0.weight.data.mul_(end_mask)
#         # m0.bias.data.mul_(end_mask)
#
#         # m0.weight.grad.data[index] = 0
#         # m0.bias.grad.data[index] = 0
#
#         # bn_mask = bn_mask.long()
#         # m0.weight.data.mul_(bn_mask)
#         # m0.bias.data.mul_(bn_mask)
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


def test():  # 79.35t

    classes = [0, 10]  # 74.10-84.60
    _, test_loader = get_train_and_test_dataloader(classes)

    # for name, m0 in model.named_modules():
    #     if isinstance(m0, nn.Linear):
    #         c = m0.weight.data.clone()
    #         d = m0.bias.data.clone()

    model.eval()
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)  # outputs = (128,20)
            # outputs = load.classify(imgs)

            # old_predicts = load.classify(imgs)
            # outputs = model(imgs)
            # predicts = torch.max(outputs, dim=1)[1]
            # predicts[:10] = old_predicts

        # predicts = outputs
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    print(str(accuracy.item()))


# test()

# print(load.model)
def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1

def train():
    accuracy = 0

    for param in model.feature.parameters():
        param.requires_grad = False

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    # pg = [p for p in model.fc.parameters()]
    opt = optim.SGD(pg, lr=learning_rate, weight_decay=0.00001)

    # classify_params = list(map(id, load.model.fc.parameters()))
    # base_params = filter(lambda p: id(p) not in classify_params,
    #                      load.model.parameters())
    # opt = torch.optim.SGD([
    #     {'params': base_params},
    #     {'params': load.model.fc.parameters(), 'lr': learning_rate}], lr=learning_rate*0.001, momentum=0.9)


    # opt = optim.SGD(load.model.parameters(), lr=learning_rate, weight_decay=0)
    classes = [-1, 20]
    train_loader = get_train_refine_dataloader(classes)

    # load.model.reset_parameters()

    # adms_loss = AngularPenaltySMLoss(20, 20, loss_type='sphereface').to(device)

    for epoch in range(100):

        if epoch > 40 and epoch < 48:
            for p in opt.param_groups:
                p['lr'] = learning_rate / (1.0 + 0.21 * (epoch - 40))
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

            # new_index = torch.nonzero(target > 19).squeeze(1)
            # old_index = torch.nonzero(target < 20).squeeze(1)

            enable_amp = True if "cuda" in device.type else False
            scaler = amp.GradScaler(enabled=enable_amp)

            new_num = numclass - 10
            old_index = torch.nonzero(target < new_num).squeeze(1)

            with amp.autocast(enabled=enable_amp):
                outputs = self.model(images)
                num_old_classes = self.numclass - 10
                T = 2
                alpha = 0.24  # 0.25
                ref_outputs = self.old_model(images)
                loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1), \
                                       F.softmax(ref_outputs.detach() / T,
                                                 dim=1)) * T * T * alpha * num_old_classes

                new = outputs[old_index, new_num:]
                old = outputs[old_index, :new_num]
                new_pre_max_value = torch.max(new, dim=1)[0]
                old_pre_max_value = torch.max(old, dim=1)[0]
                # margin should tend to be smaller
                margin = new_pre_max_value - old_pre_max_value
                sort_margin = torch.sort(margin)[0]
                idx_start = torch.floor(torch.tensor(sort_margin.shape[0] * 0.72)).long()  # 0.72 is best
                mm = sort_margin[idx_start:]

                scaled_mean = torch.mean(mm) / 20
                margin_loss = torch.log(1.6 + torch.clamp(scaled_mean, min=-1.17))

                loss2 = nn.CrossEntropyLoss()(outputs, target)
                loss_value = loss1 + loss2 + 0.36 * margin_loss  # 0.27

            opt.zero_grad()
            scaler.scale(loss_value).backward()  # backward

            # updateBN(load.model.feature)

            scaler.step(opt)  # optimize
            scaler.update()

            if step and step % 10 == 0:
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
                }, "refine.pth.tar")


train()


# shape = int(out.shape[0])
# out = out.cpu().numpy()
# d = np.linalg.norm(out, ord=2, axis=0)
# d = d + [1e-7, ]
# d = np.tile(d, (shape, 1))
# out = out / d
# out = torch.tensor(out).to(device, dtype=torch.float32)
# loss_value = adms_loss(output, target)