import torch.nn as nn
from torch.nn import functional as F
import math

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.up_branch = False
        self.getValue = False
        self.only_down = False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializait

    def forward(self, x, fusion_material_from_downBranch=None, a=0.5, once=False):
        if self.up_branch:  # enable dual branch
            x = self.feature.forward(x, fusion_material_from_downBranch, a, once=False)
            # x = self.fc(F.normalize(feat, p=2, dim=1))
            # return feat, x
        elif self.getValue:  # extract fusion information of auxiliary branch
            return self.feature.forward(x, getValue=True)
        else:   # It is acess for both main and auxiliary branches. First, the auxiliary branch is called
            # during incremental training, and can be used for exemplar saving.
            # Second, it can be called from a single-branch enabled during basic training.
            # Third, it can be used to test the main branch after incremental training.
            if self.only_down:
                return self.feature.forward(x, once=True, getValue=False)
            else:
                x = self.feature.forward(x, once=True, getValue=False)

        out = self.fc(F.normalize(x, p=2, dim=1))

        if self.up_branch:
            return x, out
        else:
            return out

    def Incremental_learning(self, numclass):

        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        # self.fc = nn.utils.weight_norm(nn.Linear(in_feature, numclass))
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

        # weight = self.weight.data
        # self.weight = Parameter(torch.Tensor(numclass, self.feature.fc.in_features))
        # out_feature = numclass - 10
        # self.weight.data[:out_feature] = weight

    def feature_extractor(self, inputs):
        return self.feature(inputs)
