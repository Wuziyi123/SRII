import torch as t
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    def __init__(self, cls_num, featur_num):
        super().__init__()

        self.cls_num = cls_num
        self.featur_num = featur_num
        self.center = nn.Parameter(t.rand(cls_num, featur_num))

    def forward(self, xs, ys):  # xs=feature,ys=target
        # xs = F.normalize(xs)
        self.center_exp = self.center.index_select(dim=0, index=ys.long())
        count = t.histc(ys, bins=self.cls_num, min=0, max=self.cls_num - 1)
        self.count_dis = count.index_select(dim=0, index=ys.long()) + 1
        loss = t.mean(t.sum((xs - self.center_exp) ** 2, dim=1) / 2.0 / self.count_dis.float())

        return loss
