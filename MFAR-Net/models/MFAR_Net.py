import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .backbone import build_backbone
from .encoder import build_encoder
from .decoder import build_decoder
from .matcher import build_matcher
from .criterion import build_criterion
from .postprocessor import build_postprocessor


class mfar_net(nn.Module):

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


def build_model(args, device):

    backbone = build_backbone(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    matcher = build_matcher(args)

    model = mfar_net(backbone, encoder, decoder, args.multi_scale)

    weight_dict = {'loss_vfl': args.vfl_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_iou': args.iou_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['vfl', 'boxes']
    criterion = build_criterion(num_classes=args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                iou_type=args.iou_type, eos_coef=args.eos_coef, losses=losses)

    criterion.to(device)
    postprocessors = build_postprocessor(args)

    return model, criterion, postprocessors
