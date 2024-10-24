from pathlib import Path
import torch
import torch.utils.data
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints
from typing import Any, Dict
import torchvision.transforms.v2 as T
import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import cv2


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(target['boxes'],
                                                     format=datapoints.BoundingBoxFormat.XYXY,
                                                     spatial_size=img.size)  # h w
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {"boxes": boxes, "labels": classes, "image_id": image_id}

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class ConvertBox(T.Transform):
    _transformed_types = (datapoints.BoundingBox,)

    def __init__(self, in_fmt="", out_fmt='', normalize=False) -> None:
        super().__init__()
        self.in_fmt = in_fmt
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'xywh': datapoints.BoundingBoxFormat.XYWH,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]):
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            # in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=self.in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt


class ResizeAndPad(object):
    def __init__(self, size, pad_value=0):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, input):
        (img, anns) = input
        w, h = img.size
        boxes_origin = []
        clsses_origin = []
        anns_new = {"boxes": anns["boxes"], "labels": anns["labels"], "image_id": anns["image_id"],
                    "area": anns["area"], "iscrowd": anns["iscrowd"]}
        if anns is not None:
            for box, cls in zip(anns["boxes"], anns["labels"]):
                boxes_origin.append(box.tolist())
                clsses_origin.append(str(cls.tolist()))
        image = np.array(img)
        transformed_img, transformed_bboxes = self.preprocess_image_and_labels(image, boxes_origin, new_shape=self.size)

        img = Image.fromarray(transformed_img)
        if anns is not None:
            for i, transformed_bbox in enumerate(transformed_bboxes):
                anns_new["boxes"][i] = torch.Tensor(transformed_bbox)
                anns_new["labels"][i] = torch.Tensor([int(clsses_origin[i])])
            anns_new["boxes"].spatial_size = self.size
            anns_new["orig_size"] = torch.Tensor(self.size)
            anns_new["size"] = torch.Tensor(self.size)
        return img, anns_new

    def preprocess_image_and_labels(self, image, bboxes, **kwargs):
        # 使用letterbox函数处理图像
        image, ratio, pad = self.letterbox(image, **kwargs)

        # 更新边界框
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox

            # 根据缩放比例调整边界框坐标和尺寸
            new_xmin = ratio[0] * xmin
            new_ymin = ratio[1] * ymin
            new_xmax = ratio[0] * xmax
            new_ymax = ratio[1] * ymax

            # 考虑填充影响，加上偏移量
            new_xmin += pad[0]
            new_ymin += pad[1]
            new_xmax += pad[0]
            new_ymax += pad[1]

            # 更新边界框
            bbox[0] = new_xmin
            bbox[1] = new_ymin
            bbox[2] = new_xmax
            bbox[3] = new_ymax
        return image, bboxes

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


def make_coco_transforms(image_set, args):
    xyxy2cxcywh = ConvertBox(in_fmt="xyxy", out_fmt="cxcywh", normalize=True)
    xyxy2xywh = ConvertBox(in_fmt="xyxy", out_fmt="xywh", normalize=True)
    xywh2cxcywh = ConvertBox(in_fmt="xywh", out_fmt="cxcywh", normalize=True)
    basic_tranform = ResizeAndPad(args.imgsize)
    if image_set == 'train':
        return T.Compose([
                          # basic_tranform,
                          T.Resize(args.imgsize),
                          T.RandomPhotometricDistort(p=0.8),
                          T.RandomZoomOut(fill=0),
                          T.RandomIoUCrop(),
                          T.SanitizeBoundingBox(),
                          T.RandomHorizontalFlip(),
                          T.Resize(args.imgsize),
                          T.ToImageTensor(),
                          T.ConvertImageDtype(),
                          T.SanitizeBoundingBox(),
                          xyxy2cxcywh   # coco:xywh, ConvertCocoPolysToMask:xywh->xyxy, xyxy->cxcywh
                          ])

    if image_set == 'val':
        return T.Compose([
                          # basic_tranform,
                          T.Resize(args.imgsize),
                          T.ToImageTensor(),
                          T.ConvertImageDtype(),  # coco:xywh, ConvertCocoPolysToMask:xywh->xyxy, postprocessor:cxcywh->xywh; bs: xyxy->xywh
                          ])
    if image_set == 'test':
        return T.Compose([
                          # basic_tranform,
                          T.Resize(args.imgsize),
                          T.ToImageTensor(),
                          T.ConvertImageDtype(),  # coco:xywh, ConvertCocoPolysToMask:xywh->xyxy, postprocessor:cxcywh->xywh; bs: xyxy->xywh
                          ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {"train": (root / "images", root / "annotations" / 'train.json'),
             "val": (root / "images", root / "annotations" / 'val.json'),
             "test": (root / "images", root / "annotations" / 'test.json')
             }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args))
    return dataset
