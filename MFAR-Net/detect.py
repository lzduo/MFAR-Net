import argparse
import time
from pathlib import Path
import numpy as np
import torch
from models.mydetr import build_model
from PIL import Image
import os
from torchvision.ops.boxes import batched_nms
from utils.distributed_utils import select_device
import torchvision.transforms.v2 as T
import cv2
import matplotlib.pyplot as plt
from utils.cam_utils import GradCAM, show_cam_on_image


# -------------------------------------------------------------------------设置参数
def get_args_parser():
    parser = argparse.ArgumentParser('Setting', add_help=False)
    parser.add_argument('--output_dir', default='./runs/v30_test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='0',
                        help='device to use for training / testing')
    parser.add_argument('--ema', default=True, type=bool,
                        help="")

    # Model parameters--------------------------------------------------------------------------------------------------
    parser.add_argument('--num_classes', type=int, default=20,
                        help="Number of object classes")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--return_idx', default=[1, 2, 3], type=list,
                        help="")
    parser.add_argument('--freeze_convs', default=True, type=bool,
                        help="")
    parser.add_argument('--freeze_norm', default=True, type=bool,
                        help="")
    parser.add_argument('--unfreeze_at', default=100, type=int,
                        help="")
    parser.add_argument('--pretrained', default="./weights/ResNet50.pth", type=str,
                        help="")
    # * Transformer
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0., type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")

    # Loss--------------------------------------------------------------------------------------------------------------
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--eos_coef', default=1e-4, type=float,
                        help="")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_iou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--use_focal_loss', default=True, type=bool,
                        help="use focal loss")
    # * Loss coefficients
    parser.add_argument('--iou_type', default="SIoU", type=str,  # <------------------------IOU
                        help="")
    parser.add_argument('--vfl_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--iou_loss_coef', default=2, type=float)

    # dataset parameters------------------------------------------------------------------------------------------------
    parser.add_argument('--data', default="../datasets/test/images")  # <--------------dataset
    parser.add_argument('--imgsize', default=[640, 640])
    parser.add_argument('--multi_scale', default=None)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='./weights/v30_dior_ultra.pth', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters-----------------------------------------------------------------------------------
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# dior classes
CLASSES = ["airplane",
           "airport",
           "baseballfield",
           "basketballcourt",
           "bridge",
           "chimney",
           "dam",
           "Expressway-Service-area",
           "Expressway-toll-station",
           "golffield",
           "groundtrackfield",
           "harbor",
           "overpass",
           "ship",
           "stadium",
           "storagetank",
           "tenniscourt",
           "trainstation",
           "vehicle",
           "windmill"
           ]
# hrrsd classes
# CLASSES = ["bridge",
#            "airplane",
#            "ground track field",
#            "vehicle",
#            "parking lot",
#            "T junction",
#            "baseball diamond",
#            "tennis court",
#            "basketball court",
#            "ship",
#            "crossroad",
#            "harbor",
#            "storage tank"
#            ]
# nwpu classes
# CLASSES = ["airplane",
#            "ship",
#            "storage_tank",
#            "baseball_diamond",
#            "tennis_court",
#            "basketball_court",
#            "ground_track_field",
#            "harbor",
#            "bridge",
#            "vehicle"
#            ]
# CLASSES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
#            'tennis-court',
#            'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
#            'helicopter', 'container-crane'
#            ]

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]


def box_cxcywh_to_xyxy(x):
    # 将DETR的检测框坐标(x_center,y_cengter,w,h)转化成coco数据集的检测框坐标(x0,y0,x1,y1)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    # 把比例坐标乘以图像的宽和高，变成真实坐标
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    # b = out_bbox
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_boxes(scores, boxes, confidence=0.75, apply_nms=False, iou=0.25):
    # 筛选出置信度高的框
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    return scores, boxes


def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # 把检测框画到图片上
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = colors[color]
    x = [i * 800 / 640 for i in x]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def main(args):
    print(args)
    device = torch.device(select_device(args.device, batch_size=1))

    model, criterion, postprocessors = build_model(args, device)
    model.eval()
    criterion.eval()
    checkpoint = torch.load(args.resume, map_location='cuda')
    if args.ema:
        model.load_state_dict(checkpoint['ema']['module'])
    else:
        model.load_state_dict(checkpoint['model'])

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", n_parameters)

    images_path = os.listdir(args.data)
    target_layers = [model.encoder.pan_blocks[-1], model.encoder.pan_blocks[-2], model.encoder.fpn_blocks[0]]

    cam_extractor = GradCAM(model, target_layers)
    for image_item in images_path:
        print("inference_image:", image_item)
        image_path = os.path.join(args.data, image_item)

        test_transforms = T.Compose([T.Resize(args.imgsize),
                                     T.ToImageTensor(),
                                     T.ConvertImageDtype(torch.float32),
                                     ])

        image = Image.open(image_path)
        image_tensor = test_transforms(image)
        image_tensor = image_tensor.to(device)
        time1 = time.time()
        inference_result = model(image_tensor.unsqueeze(0))

        time2 = time.time()
        print("inference_time:", time2 - time1)
        probas = inference_result['pred_logits'].softmax(-1)[0, :, :].cpu()
        bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0,].cpu(),
                                       (image_tensor.shape[2], image_tensor.shape[1]))
        scores, boxes = filter_boxes(probas, bboxes_scaled)
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
        image_show = cv2.resize(np.asarray(image).copy(), [800, 800])
        for i in range(boxes.shape[0]):
            class_id = scores[i].argmax()
            label = CLASSES[class_id]
            confidence = scores[i].max()
            text = f"{label} {confidence:.3f}"  # <--------------label
            text = None
            plot_one_box(boxes[i], image_show, label=text, color=class_id)

            grayscale_cam = cam_extractor(input_tensor=image_tensor.unsqueeze(0), target_category=[class_id])
            grayscale_cam = grayscale_cam[0, :]
            image_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()  # 转置为 (H, W, C) 格式，便于显示
            visualization = show_cam_on_image(image_numpy.astype(dtype=np.float32), grayscale_cam, use_rgb=True)
            heatmap_image = Image.fromarray(visualization)
            heatmap_image.save(os.path.join(args.output_dir, image_item + f"_heatmap{class_id}.jpg"))
        image_save = Image.fromarray(image_show)
        image_save.save(os.path.join(args.output_dir, image_item))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
