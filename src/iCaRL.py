import math
import os

import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch.cuda import amp
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim

import my_utils
from myNetwork import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader

from iCIFAR100 import *
from ResNet import ResNet

import sys

from centerloss import CenterLoss


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


sys.stdout = Logger("result.log", sys.stdout)
sys.stderr = Logger("error.log", sys.stderr)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, data_path, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate):

        super(iCaRLmodel, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.test_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                      # transforms.Resize(img_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                           (0.2675, 0.2565, 0.2761))])

        file_path = os.path.join(os.getcwd(), data_path)
        # file_path = os.path.join(os.path.dirname(__file__), 'dataset')
        self.train_dataset = iCIFAR100(file_path, transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100(file_path, test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.test_loader = None
        self.weight = None
        self.bias = None
        self.cfg_mask = []
        self.test_material = []

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        self.train_dataset.incremental = False
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def new_beforeTrain(self, i, up_model):
        self.model.eval()
        self.train_dataset.incremental = True

        branch_begin_class = i * self.task_size
        classes = [branch_begin_class, branch_begin_class + self.task_size]
        self.train_loader, self.test_loader = self._incremental_get_train_and_test_dataloader(up_model, classes, i)
        # self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        self.model.train()
        self.model.to(device)

    def _incremental_get_train_and_test_dataloader(self, up_model, classes, i):
        self.train_dataset.getDoubleBranchTrainData(up_model, classes, i)
        # self.test_dataset.getTestData(classes)
        self.test_dataset.getBranchTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    def _get_train_refine_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        return train_loader

    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1

    # train model
    # compute loss
    # evaluate model
    def train(self, incremental, iter, up_model=None, branch=False):
        accuracy = 0
        best_map = 0.0

        # centerloss = None
        # opt_center = None
        # if branch:
        #     centerloss = CenterLoss(up_model.numclass, 512).to(device)
        #     opt_center = optim.SGD(centerloss.parameters(), lr=0.5)

        opt_down = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0)

        for param in self.model.feature.parameters():
            param.requires_grad = True

        if iter == 1 and incremental:
            opt = optim.SGD(up_model.model.parameters(), lr=up_model.learning_rate, weight_decay=0.00001, momentum=0.9,
                            nesterov=True)
        elif iter == 2 and incremental:
            for param in self.model.feature.parameters():
                param.requires_grad = False
            pg = [p for p in self.model.parameters() if p.requires_grad]
            opt = optim.SGD(pg, lr=self.learning_rate, weight_decay=0.00001, momentum=0.9, nesterov=True)
            classes = [self.numclass - self.task_size, self.numclass]
            self.train_loader = self._get_train_refine_dataloader(classes)
        elif iter == 0 and incremental:
            opt = optim.SGD(self.model.parameters(), lr=2.0, weight_decay=0)
        else:
            opt = optim.SGD(self.model.parameters(), lr=2.0, weight_decay=0)

        opt.zero_grad()
        opt_down.zero_grad()

        for epoch in range(self.epochs):
            # if not branch:
            if iter == 2 and incremental:
                if epoch == 0:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 62.5
                if epoch == 60:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 125
            else:
                if epoch > 40 and epoch < 48:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / (1.0 + 0.21 * (epoch - 40))
                if epoch == 48:
                    if not incremental:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5)
                    else:
                        for p in opt.param_groups:
                            p['lr'] = self.learning_rate / 2.5
                elif epoch == 62:
                    if not incremental:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25)
                    else:
                        for p in opt.param_groups:
                            p['lr'] = self.learning_rate / 12.5
                elif epoch == 80:
                    if not incremental:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125)
                    else:
                        for p in opt.param_groups:
                            p['lr'] = self.learning_rate / 62.5
            # else:
            #     if epoch == 48:
            #         for p in opt.param_groups:
            #             p['lr'] = up_model.learning_rate / 2.5
            #     elif epoch == 62:
            #         for p in opt.param_groups:
            #             p['lr'] = up_model.learning_rate / 12.5
            #     elif epoch == 80:
            #         for p in opt.param_groups:
            #             p['lr'] = up_model.learning_rate / 12.5

            if branch:
                if epoch > 40 and epoch < 48:
                    for p in opt_down.param_groups:
                        p['lr'] = self.learning_rate / (1.0 + 0.15 * (epoch - 40))
                if epoch == 48:
                    # opt_down = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5)
                    for p in opt_down.param_groups:
                        p['lr'] = self.learning_rate / 2.5
                elif epoch == 62:
                    # opt_down = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25)
                    for p in opt_down.param_groups:
                        p['lr'] = self.learning_rate / 12.5
                elif epoch == 80:
                    # opt_down = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125)
                    for p in opt_down.param_groups:
                        p['lr'] = self.learning_rate / 62.5

            # print(self.model.feature)
            if not branch:
                model = self.model.feature
            else:
                model = up_model.model.feature

            if not branch:  # Basic training without auxiliary branches enabled
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    images, target = images.to(device), target.to(device)
                    enable_amp = True if "cuda" in device.type else False
                    scaler = amp.GradScaler(enabled=enable_amp)

                    if incremental and not iter:

                        if epoch == 0 and step == 0:
                            print("record guarantees that it is the last time of sparsification and can record the pre-crop result!")
                            total = 0
                            for m in model.modules():
                                if isinstance(m, nn.BatchNorm2d):
                                    total += m.weight.data.shape[0]

                            bn = torch.zeros(total)
                            index = 0
                            for m in model.modules():
                                if isinstance(m, nn.BatchNorm2d):
                                    size = m.weight.data.shape[0]
                                    bn[index:(index + size)] = m.weight.data.abs().clone()
                                    index += size

                            y, i = torch.sort(bn)
                            thre_index = int(total * 0.5)
                            thre = y[thre_index]

                            pruned = 0
                            cfg = []
                            cfg_mask = []
                            for k, m in enumerate(model.modules()):
                                if isinstance(m, nn.BatchNorm2d):
                                    weight_copy = m.weight.data.clone()
                                    mask = weight_copy.abs().gt(thre).float().cuda()
                                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                                    cfg.append(int(torch.sum(mask)))
                                    cfg_mask.append(mask.clone())
                                    # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                                    #       format(k, mask.shape[0], int(torch.sum(mask))))
                                elif isinstance(m, nn.MaxPool2d):
                                    cfg.append('M')

                            pruned_ratio = pruned / total
                            print('Pre-processing Successful!')
                            self.cfg_mask = cfg_mask

                        #  end of if
                        cfg_mask = self.cfg_mask
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
                    # end of if incremental and not iter:


                    # old_num = self.numclass - 11
                    # new_index = torch.nonzero(target > old_num).squeeze(1)
                    new_num = self.numclass - 10
                    old_index = torch.nonzero(target < new_num).squeeze(1)
                    with amp.autocast(enabled=enable_amp):
                        if incremental and iter == 1:
                            pass
                        elif incremental and iter == 2:
                            outputs = self.model(images)
                            num_old_classes = self.numclass - 10
                            T = 2
                            alpha = 0.25  # 0.25
                            ref_outputs = self.old_model(images)
                            loss1 = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                            F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * alpha

                            new = outputs[old_index, new_num:]
                            old = outputs[old_index, :new_num]
                            if new.numel() and old.numel():
                                new_pre_max_value = torch.max(new, dim=1)[0]
                                old_pre_max_value = torch.max(old, dim=1)[0]

                                margin = new_pre_max_value - old_pre_max_value
                                sort_margin = torch.sort(margin)[0]
                                idx_start = torch.floor(torch.tensor(sort_margin.shape[0] * 0.72)).long()
                                mm = sort_margin[idx_start:]

                                scaled_mean = torch.mean(mm) / 20
                                margin_loss = torch.log(1.98 + torch.clamp(scaled_mean, min=-1.17))
                            else:
                                margin_loss = torch.tensor(0)

                            loss2 = nn.CrossEntropyLoss()(outputs, target)
                            loss_value = loss1 + loss2 + 0.36 * margin_loss  # 0.27
                        else:
                            output = self.model(images)  # output = (128,30)
                            output, target = output.to(device), target.to(device)
                            loss_value = nn.CrossEntropyLoss()(output, target)

                    opt.zero_grad()
                    scaler.scale(loss_value).backward()  # backward

                    if iter == 1:
                        self.updateBN(model)
                    elif incremental and not iter:
                        # print(self.model)
                        # exit(-1)
                        cfg_mask = self.cfg_mask
                        layer_id_in_cfg = 0
                        # skip = [4, 5, 14, 15, 27, 28, 46, 47, 53, 54, 55, 56, 57, 58, 59]
                        skip = [7, 8, 12, 13, 17, 18, 20, 21, 22, 23, 24, 25, 26]
                        start_mask = torch.ones(3)
                        end_mask = cfg_mask[layer_id_in_cfg]

                        for name, m0 in self.model.named_modules():
                            # is_mask = name.endswith("1_1") or name.endswith("2_1")
                            # is_mask = name.endswith("gff") or name.endswith("2_1")
                            if isinstance(m0, nn.BatchNorm2d):
                                layer_id_in_cfg += 1
                                start_mask = end_mask.clone()
                                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                                    end_mask = cfg_mask[layer_id_in_cfg]
                            elif isinstance(m0, nn.Conv2d):
                                if layer_id_in_cfg not in skip:
                                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                                    # print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                                    m0.weight.grad.data[:, idx0, :, :].mul_(0)
                                    m0.weight.grad.data[idx1, :, :, :].mul_(0)
                        # print("")
                    scaler.step(opt)  # optimize
                    scaler.update()

                    end = len(self.train_loader) - 1

                    if epoch == 99 and step == end and incremental and iter == 0:

                        if os.path.isfile("best.pth.tar"):
                            print("=> loading checkpoint '{}'".format("best.pth.tar"))
                            checkpoint = torch.load("best.pth.tar")
                            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                        else:
                            print("=> no checkpoint found at '{}'".format("best.pth.tar"))

                        print("store temp weights!")
                        filename = "temp-base.pth.tar"
                        torch.save({
                            'state_dict': self.model.state_dict(),
                            'cfg_mask': self.cfg_mask,
                            'optimizer': opt.state_dict(),
                            'class_mean_set': self.class_mean_set,
                            'exemplar_set': self.exemplar_set,
                            'old_model': self.old_model,
                            'test_loader': self.test_loader,
                            'weight': self.weight,
                            'bias': self.bias,
                        }, filename)  # split-incremental-2.pth.tar

                    if step and step % 27 == 0:
                        print('epoch:%d,step:%d,loss:%.4f' % (epoch, step, loss_value.item()))
            else:
                for step, (indexs, images, target, up_images, up_targets) in enumerate(self.train_loader):
                    images, target = images.to(device), target.to(device)
                    up_images, up_targets = up_images.to(device), up_targets.to(device)
                    enable_amp = True if "cuda" in device.type else False
                    scaler = amp.GradScaler(enabled=enable_amp)

                    self.model.only_down = True
                    with amp.autocast(enabled=enable_amp):
                        output = self.model(images)
                        output = output.to(device)
                        loss_down = nn.CrossEntropyLoss()(output, target)
                    # loss_down.requires_grad_(True)
                    opt_down.zero_grad()
                    # loss_down.backward()
                    scaler.scale(loss_down).backward()  # backward
                    # opt_down.step()
                    self.model.only_down = False
                    scaler.step(opt_down)  # optimize
                    scaler.update()
                    # if step and step % 27 == 0:
                    #     print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_down.item()))

                    # if step == 0:
                    #     print(self.model.feature._modules["layer2"]._modules["0"].conv1.weight.grad.data)

                    self.model.eval()
                    self.model.getValue = True
                    with torch.no_grad():
                        x1n, g1, x2n, g2, x3n, g3 = self.model(images)
                        # layer1, layer2, layer3 = self.model(images)
                    self.model.train()
                    self.model.getValue = False
                    # fusion_material_from_downBranch = None

                    fusion_material_from_downBranch = [x1n, g1, x2n, g2, x3n, g3]

                    up_model.model.up_branch = True
                    # 0.4-82.35  0.3-83.85(82.8-84.9)  0.6-83.0  0.2-82.45
                    # if epoch < 81:
                    #     a = 0.3 - epoch / 267
                    # else:
                    #     a = 0
                    # a = 1
                    # a = max(0.3 - epoch / 267, 0)
                    a = max(0.1 - epoch / 800, 0)

                    # a = max(0.5 - epoch / 0.8 * 2
                    # T, 0)  # Baseline
                    # a = 0.5 * sin((2π / 2T) * epoch)  # sin
                    # a = 0.5 * cos((2π / 4T) * epoch)  # cos
                    # a = max(0.5 - (0.5 / (T ^ 2)) * epoch ^ 2, 0)  # quartic function

                    next_end = len(self.train_loader) - 2
                    if step == next_end:
                        up_model.test_material.append(a)
                    if step == 0:
                        up_model.test_material = []
                    # ***********************************************************************************
                    new_num = up_model.numclass - 10
                    old_index = torch.nonzero(up_targets < new_num).squeeze(1)
                    with amp.autocast(enabled=enable_amp):
                        feat, outputs = up_model.model(up_images, fusion_material_from_downBranch, a)
                        # feat, outputs = up_model.model(up_images, fusion_material_from_downBranch, a)
                        num_old_classes = up_model.numclass - 10
                        T = 2
                        alpha = 0.25  # 0.25
                        ref_outputs = up_model.old_model(up_images)
                        loss1 = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                        F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * alpha

                        new = outputs[old_index, new_num:]
                        old = outputs[old_index, :new_num]
                        if new.numel() and old.numel():
                            new_pre_max_value = torch.max(new, dim=1)[0]
                            old_pre_max_value = torch.max(old, dim=1)[0]

                            margin = new_pre_max_value - old_pre_max_value
                            sort_margin = torch.sort(margin)[0]
                            idx_start = torch.floor(torch.tensor(sort_margin.shape[0] * 0.72)).long()
                            mm = sort_margin[idx_start:]

                            scaled_mean = torch.mean(mm) / 20
                            margin_loss = torch.log(1.98 + torch.clamp(scaled_mean, min=-1.17))
                        else:
                            margin_loss = torch.tensor(0)

                        loss2 = nn.CrossEntropyLoss()(outputs, up_targets)
                        # center_loss = centerloss(feat, up_targets)
                        # 0.3-84.5 0.32-
                        # print(loss1,loss2)
                        loss_up = loss1 + loss2 + 0.36 * margin_loss  # 0.29-0.13

                    opt.zero_grad()
                    # opt_center.zero_grad()
                    scaler.scale(loss_up).backward()  # backward
                    up_model.updateBN(model)

                    scaler.step(opt)  # optimize
                    # scaler.step(opt_center)
                    scaler.update()

                    up_model.model.up_branch = False

                    end = len(self.train_loader) - 1
                    if epoch == 99 and step == end and incremental and iter == 1:

                        if os.path.isfile("best.pth.tar"):
                            print("=> loading checkpoint '{}'".format("best.pth.tar"))
                            checkpoint = torch.load("best.pth.tar")
                            up_model.model.load_state_dict(checkpoint['state_dict'], strict=False)
                        else:
                            print("=> no checkpoint found at '{}'".format("best.pth.tar"))

                        print("store merge weights!")
                        filename = "temp-merge-" + str(incremental) + ".pth.tar"
                        torch.save({
                            'state_dict': up_model.model.state_dict(),
                            'cfg_mask': up_model.cfg_mask,
                            'optimizer': opt.state_dict(),
                            'class_mean_set': up_model.class_mean_set,
                            'exemplar_set': up_model.exemplar_set,
                            'old_model': up_model.old_model,
                            'weight': up_model.weight,
                            'bias': up_model.bias,
                            'down_branch': self.model.state_dict()
                        }, filename)  # split-incremental-2.pth.tar
                        # if incremental == 2:
                        #     exit(-1)

                    if step and step % 27 == 0:
                        print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_up.item()))

            #  end of else

            if branch:
                accuracy = up_model._test(self.test_loader, 1, branch=branch, down_model=self.model)
                print('epoch:%d,accuracy_up:%.3f,' % (epoch, accuracy))
            else:
                accuracy = self._test(self.test_loader, 1)
                print('epoch:%d,accuracy:%.3f,' % (epoch, accuracy))

            # update best model, store begin-limit is 50.0%.
            if accuracy > best_map:
                best_map = accuracy

            # only save best weights every iter
            if best_map == accuracy:
                filename = "best.pth.tar"
                print("store best weights!")
                if not branch:
                    torch.save({
                        'state_dict': self.model.state_dict(),
                    }, filename)  # split-incremental-2.pth.tar
                else:
                    torch.save({
                        'state_dict': up_model.model.state_dict(),
                        'down_branch': self.model.feature.state_dict()
                    }, filename)  # split-incremental-2.pth.tar

        if os.path.isfile("best.pth.tar"):
            print("=> loading checkpoint '{}'".format("best.pth.tar"))
            checkpoint = torch.load("best.pth.tar")
            if not branch:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                up_model.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("=> no checkpoint found at '{}'".format("best.pth.tar"))

        if branch:
            return accuracy, self.model
        else:
            return accuracy

    def _test(self, testloader, mode, branch=False, down_model=None):
        if mode == 0:
            print("compute NMS")
        self.model.eval()

        if down_model:
            down_model.eval()

        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            if not branch:
                with torch.no_grad():
                    outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            else:
                down_model.getValue = True
                self.model.up_branch = True
                with torch.no_grad():
                    x1n, g1, x2n, g2, x3n, g3 = down_model(imgs)
                    material_from_down = [x1n, g1, x2n, g2, x3n, g3]
                    _, outputs = self.model(imgs, material_from_down, self.test_material[0])
                    # feat, outputs = self.model(imgs, material_from_down, self.test_material[0])
                self.model.up_branch = False
                down_model.getValue = False
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs

            # if branch:
            #     with torch.no_grad():
            #         new_num = outputs.shape[1] - 11
            #         new_index = torch.nonzero(predicts > new_num).squeeze(1)
            #         new_imgs = imgs[new_index, ::]
            #         outputs_new = down_model(new_imgs)
            #     predicts_new = torch.max(outputs_new, dim=1)[1]
            #     new_start = int(outputs.shape[1] - 10)
            #     predicts_new = torch.add(predicts_new, new_start)
            #     predicts[new_index] = predicts_new

            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()

        if down_model:
            down_model.train()

        return accuracy

    def _compute_loss(self, indexs, imgs, target, branch=False):

        output = self.model(imgs)  # output = (128,10)

        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if not branch and self.old_model:
            # old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_output = self.old_model(imgs)
            old_target = torch.sigmoid(old_output)
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            return F.binary_cross_entropy_with_logits(output, target)

    # change the size of examplar
    def afterTrain(self, accuracy, branch=False, down_model=None):
        self.model.eval()
        m = int(self.memory_size / self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass - self.task_size, self.numclass):
            print('construct class %s examplar:' % (i), end='')  # 10-20
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m)
        self.numclass += self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        if branch:
            exit(-1)
            KNN_accuracy = self._test(self.test_loader, 1, branch=branch, down_model=down_model)
        else:
            KNN_accuracy = self._test(self.test_loader, 1)
        print("NMS accuracy：" + str(KNN_accuracy.item()))
        root_path = os.path.join(os.path.dirname(__file__), '../model/')
        filename = root_path + 'accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))

        for i in range(m):
            # shape：batch_size*512    (512,)-( ((1,512)+(500,512))/1..)
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size    x = (500,512)
            x = np.linalg.norm(x, axis=1)  # x = (500,)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        #  exemplar = List[(32,32,3)*200]
        self.exemplar_set.append(exemplar)
        # self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        #  x =(500,3,32,32)   images = (500,32,32,3)  feature_extractor_output = (500,512) class_mean =(512)

        with torch.no_grad():
            out = self.model.feature_extractor(x).detach()
        feature_extractor_output = F.normalize(out).cpu().numpy()
        # feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s" % (str(index)))
            exemplar = self.exemplar_set[index]
            # exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean(exemplar, self.classify_transform)
            class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2
            #  class_mean = List[(512,)*10 ]
            self.class_mean_set.append(class_mean)


    def classify(self, test):
        result = []
        with torch.no_grad():
            out = self.model.feature_extractor(test).detach()
        test = F.normalize(out).cpu().numpy()
        # test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
