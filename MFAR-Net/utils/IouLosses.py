import math

import torch
from torch import nn


class IouLoss(nn.Module):
    ''' :param monotonous: {
            None: origin
            True: monotonic FM
            False: non-monotonic FM
        }'''
    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype='SIoU', monotonous=False):
        super().__init__()
        assert hasattr(self, f'_{ltype}'), f'The loss function {ltype} does not exist'
        self.ltype = ltype
        self.monotonous = monotonous
        self.iou_mean = nn.Parameter(torch.tensor(1.))

    def forward(self, pred, target, ret_iou=False, scale=False):
        self.pred = pred
        self.target = target

        # 计算预测框和目标框的相关信息
        self.compute_box_info()

        # 计算损失
        loss, iou = getattr(self, f'_{self.ltype}')()

        # 根据 monotonous 参数对损失进行缩放
        if scale:
            loss = self._scaled_loss(loss, iou)

        # 训练阶段更新 iou_mean
        if self.training:
            self.iou_mean.data.mul_(1 - self.momentum)
            self.iou_mean.data.add_(self.momentum * self.iou.detach().mean())

        return [loss, iou] if ret_iou else loss

    def compute_box_info(self):
        # x,y,w,h
        self.pred_xy = (self.pred[..., :2] + self.pred[..., 2: 4]) / 2
        self.pred_wh = self.pred[..., 2: 4] - self.pred[..., :2]
        self.target_xy = (self.target[..., :2] + self.target[..., 2: 4]) / 2
        self.target_wh = self.target[..., 2: 4] - self.target[..., :2]
        # x0,y0,x1,y1
        self.min_coord = torch.min(self.pred[:, None, :4], self.target[..., :4])
        self.max_coord = torch.max(self.pred[:, None, :4], self.target[..., :4])
        # The overlapping region
        self.wh_inter = torch.relu(self.min_coord[..., 2: 4] - self.max_coord[..., :2])
        self.s_inter = torch.prod(self.wh_inter, dim=-1)
        # The area covered
        self.s_union = torch.prod(self.pred_wh, dim=-1)[:, None] + torch.prod(self.target_wh, dim=-1) - self.s_inter
        # The smallest enclosing box
        self.wh_box = self.max_coord[..., 2: 4] - self.min_coord[..., :2]
        self.s_box = torch.prod(self.wh_box, dim=-1)
        self.l2_box = torch.square(self.wh_box).sum(dim=-1)
        # The central points' connection of the bounding boxes
        self.d_center = self.pred_xy[:, None] - self.target_xy
        self.l2_center = torch.square(self.d_center).sum(dim=-1)
        # IoU
        self.iou = self.s_inter / self.s_union
        # IoU loss
        self.iou_loss = 1 - self.s_inter / self.s_union

    def _scaled_loss(self, loss, iou=None):
        if isinstance(self.monotonous, bool):
            beta = (self.iou_loss.detach() if iou is None else iou) / self.iou_mean

            if self.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss *= beta / divisor
        return loss

    def _IoU(self):
        return self.iou_loss, self.iou

    def _WIoU(self):
        dist = torch.exp(self.l2_center / self.l2_box.detach())
        return dist * self.iou_loss, dist * self.iou

    def _EIoU(self):
        penalty = self.l2_center / self.l2_box + torch.square(self.d_center / self.wh_box).sum(dim=-1)
        return self.iou_loss + penalty, self.iou - penalty

    def _GIoU(self):
        return self.iou_loss + (self.s_box - self.s_union) / self.s_box,\
               self.iou - (self.s_box - self.s_union) / self.s_box

    def _DIoU(self):
        return self.iou_loss + self.l2_center / self.l2_box, self.iou - self.l2_center / self.l2_box

    def _CIoU(self, eps=1e-4):
        v = 4 / math.pi ** 2 * \
            (torch.atan(self.pred_wh[..., 0] / (self.pred_wh[..., 1] + eps)) -
             torch.atan(self.target_wh[..., 0] / (self.target_wh[..., 1] + eps))) ** 2
        alpha = v / (self.iou_loss + v)
        return self.iou_loss + self.l2_center / self.l2_box + alpha.detach() * v, \
               self.iou - self.l2_center / self.l2_box - alpha.detach() * v

    def _SIoU(self, theta=4):
        # Angle Cost
        angle = torch.arcsin(torch.abs(self.d_center).min(dim=-1)[0] / (self.l2_center.sqrt() + 1e-4))
        angle = torch.sin(2 * angle) - 2
        # Dist Cost
        dist = angle[..., None] * torch.square(self.d_center / self.wh_box)
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        # Shape Cost
        d_shape = torch.abs(self.pred_wh[:, None, :] - self.target_wh)
        big_shape = torch.max(self.pred_wh[:, None, :], self.target_wh)
        w_shape = 1 - torch.exp(- d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(- d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape ** theta + h_shape ** theta
        return self.iou_loss + (dist + shape) / 2, self.iou - (dist + shape) / 2

    def __repr__(self):
        return f'{self.__class__.__name__}(iou_mean={self.iou_mean.item():.3f})'


if __name__ == '__main__':
    def xywh2xyxy(labels, i=0):
        labels = labels.clone()
        labels[..., i:i + 2] -= labels[..., i + 2:i + 4] / 2
        labels[..., i + 2:i + 4] += labels[..., i:i + 2]
        return labels


    torch.manual_seed(0)
    iouloss = IouLoss(ltype='SIoU').requires_grad_(False)
    pred = torch.tensor([[1, 20, 30, 40]])
    gt = torch.tensor([[1, 19, 30, 40]])
    loss = iouloss(pred, gt, ret_iou=True, scale=False)
    print(loss)
