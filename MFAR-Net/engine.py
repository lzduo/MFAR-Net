# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import math
import sys
from typing import Iterable
import cv2
import torch
import numpy as np
from utils import misc
from data.coco_eval import CocoEvaluator
from utils.map import MeanAveragePrecison


def draw_bboxes_on_image(image, bboxes, classes, colors=None):
    """
    绘制边界框到图像上。

    Args:
        image (np.ndarray): 原始图像，形状为(H, W, C)，其中H为高度，W为宽度，C为通道数（通常为3）。
        bboxes (List[List[float]]): 边界框坐标列表，每个边界框由[x_min, y_min, x_max, y_max]表示。
        classes (List[int]): 目标类别的整数列表，与边界框一一对应。
        colors (List[Tuple[int, int, int]], optional): 边界框颜色列表，与类别对应。若未提供，将自动生成颜色。

    Returns:
        np.ndarray: 绘制好边界框的图像。
    """
    if colors is None:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(classes))]

    image = image.copy()
    for bbox, cls, color in zip(bboxes, classes, colors):
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

    return image

def coco_ann_to_bbox_and_classes(coco_ann):
    """
    将COCO格式的单个标注转换为边界框坐标和类别列表。

    Args:
        coco_ann (dict): COCO格式的单个标注字典。

    Returns:
        Tuple[List[List[float]], List[int]]:
            - 边界框坐标列表，每个边界框由[x_min, y_min, x_max, y_max]表示。
            - 类别列表，与边界框一一对应。
    """
    bboxes = []
    classes = []

    anns = coco_ann.get('annotations', [])
    for ann in anns:
        bbox = ann['bbox']
        bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        classes.append(ann['category_id'])

    return bboxes, classes


def train_one_epoch(model: torch.nn.Module, cam_loss_generator, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, output_dir,
                    device: torch.device, epoch: int, max_norm: float = 0, ema=None, with_cam_loss=False):
    model.train()
    criterion.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        model.train()
        
        # images = samples.cpu().numpy().transpose((0, 2, 3, 1))  # 调整通道至最后

        # 遍历每个样本及其对应的标注
        # for i, (image, target) in enumerate(zip(images, targets)):
        #     # 提取COCO格式标注中的边界框和类别
        #     bboxes, classes = coco_ann_to_bbox_and_classes(target)
        #
        #     # 绘制边界框
        #     image_with_boxes = draw_bboxes_on_image(image, bboxes, classes)
        #
        #     # 保存带框图片
        #     save_path = os.path.join(output_dir, "check_status")
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     cv2.imwrite(os.path.join(save_path, f"annotated_image_{epoch}_{i}.jpg"), image_with_boxes)
            
        samples = samples.to(device)
        # print(type(samples))
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if with_cam_loss:
            cam_loss_generator.clean_list()
        outputs = model(samples, targets)

        if with_cam_loss:
            cam_loss = cam_loss_generator(
                                          # model,
                                          samples,
                                          outputs,
                                          targets)

        model.train()
        loss_dict = criterion(outputs, targets)

        if with_cam_loss:
            loss_dict["cam_loss"] = cam_loss * 8

        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name, '      ', param.grad)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = misc.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, num_classes):
    model.eval()
    criterion.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    print_freq = 100
    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = [postprocessors.iou_types]
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    # map_calculator = MeanAveragePrecison(device=device)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = misc.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # print("orig_target_sizes", orig_target_sizes)
        results = postprocessors(outputs, orig_target_sizes)
        # for result, target in zip(results, targets):
        #     labels = np.array(result["labels"].tolist()).reshape(-1, 1)
        #     boxes = np.array(result["boxes"].tolist())
        #     scores = np.array(result["scores"].tolist()).reshape(-1, 1)
        #     result_transformed = np.column_stack((boxes, scores, labels))
        #     labels_t = np.array(target["labels"].tolist()).reshape(-1, 1)
        #     boxes_t = np.array(target["boxes"].tolist())
        #     target_transformed = np.column_stack((labels_t, boxes_t))
        #     map_calculator.process_batch(detections=result_transformed, labels=target_transformed)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res, output_dir)
    # ap = map_calculator.calculate_ap_per_class()
    # mAP_at_iou_0_5 = np.mean(ap[:, 0])
    # print(f"mAP at IoU 0.5: {mAP_at_iou_0_5:.4f}")
    # mAPs = []
    # for iou in range(ap[-1].size):
    #     mAP = np.mean(ap[:, iou])  # 计算所有类别在当前 IoU 时的 AP 平均值
    #     mAPs.append(mAP)
    #
    # mean_mAP_over_range = np.mean(mAPs)  # 计算所有 IoU 置信度范围内 mAP 的平均值
    # print(f"Mean mAP over IoU range {0.5}:{0.95}: {mean_mAP_over_range:.4f}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_stats, print_coco = summarize(coco_evaluator.coco_eval["bbox"])
        voc_map_info_list = []
        classes = [v for v in range(num_classes)]
        for i in range(len(classes)):
            stats, _ = summarize(coco_evaluator.coco_eval["bbox"], catId=i)
            voc_map_info_list.append(" {:15}: {}".format(classes[i], stats[1]))

        print_voc = "\n".join(voc_map_info_list)
        print(print_voc)

        # 将验证结果保存至txt文件中
        with open(os.path.join(output_dir, "record_mAP_.txt"), "w") as f:
            record_lines = ["COCO results:",
                            print_coco,
                            "",
                            "mAP(IoU=0.5) for each category:",
                            print_voc]
            f.write("\n".join(record_lines))
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator, {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info
