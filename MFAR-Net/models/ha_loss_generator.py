import time
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.cam_utils import GradCAM


class HALossGenerator(nn.Module):
    def __init__(self, model, target_layers, model_type, checkpoint, device):
        super().__init__()
        self.cam = GradCAM(model=model, target_layers=target_layers)
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam_model.to(device=device)
        self.sam_predictor = SamPredictor(self.sam_model)

    def cam_mask_generator(self, input_tensor, output, targets_per_img):

        cam_masks = {}
        for class_idx in targets_per_img.keys():
            grayscale_cam = self.cam(input_tensor=input_tensor.unsqueeze(0), output=output, target_category=class_idx)

            grayscale_cam = grayscale_cam[0, :]
            cam_masks[class_idx] = grayscale_cam

        return cam_masks    # dict:{"class_idx":mask, }

    def sam(self, input_img_ndarray, targets_per_img):

        self.sam_predictor.set_image(input_img_ndarray)
        sam_masks = {}
        for k, v in targets_per_img.items():
            scaled_bbox_list = [[p * int(input_img_ndarray.shape[0]) for p in bbox] for bbox in v]
            final_bbox_xyxy_list = [[
                    cx  - w  / 2,  # Xmin
                    cy  - h  / 2,  # Ymin
                    cx  + w  / 2,  # Xmax
                    cy  + h  / 2   # Ymax
                ] for cx, cy, w, h in scaled_bbox_list]
            input_boxes = torch.tensor(final_bbox_xyxy_list, device=self.sam_predictor.device)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(input_boxes, input_img_ndarray.shape[:2])
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            combined_mask = np.logical_or.reduce(masks.cpu(), axis=0)
            mask = combined_mask[0].numpy().astype(int)

            sam_masks[k] = mask
        return sam_masks

    @staticmethod
    def loss_calculator(cam_maps, sam_masks):
        total_loss_per_image = 0
        num_images = len(cam_maps)

        for img_id, cam_map in cam_maps.items():
            # 将numpy数组转为浮点类型并归一化至0-1范围内
            cam_map = np.array(cam_map, dtype=np.float32)
            # 将NaN值替换为0
            cam_map[np.isnan(cam_map)] = 0
            max_value = np.max(cam_map)
            if max_value == 0:
                max_value = 1e-7
            cam_map /= max_value

            sam_mask = np.array(sam_masks[img_id], dtype=np.float32)
            sam_mask[np.isnan(sam_mask)] = 0
            # 计算目标区域内的损失
            inter_area = np.sum(sam_mask)
            outer_area = np.sum(1 - sam_mask)

            # 防止除以零的情况
            if inter_area != 0:
                loss_inter = np.sum((1 - cam_map) * sam_mask) / inter_area
            else:
                loss_inter = np.sum((1 - cam_map) * sam_mask)

            if outer_area != 0:
                loss_outer = np.sum(cam_map * (1 - sam_mask)) / outer_area
            else:
                loss_outer = np.sum(cam_map * (1 - sam_mask))

            # 组合两部分损失
            combined_loss = loss_inter + loss_outer

            # 汇总每幅图像的损失
            total_loss_per_image += combined_loss

        # 返回平均损失
        return total_loss_per_image / num_images if num_images != 0 else total_loss_per_image

    @staticmethod
    def dice_loss(cam_maps, sam_masks):
        smooth = 1e-5
        total_loss_per_image = 0
        num_images = len(cam_maps)
        for img_id, cam_map in cam_maps.items():
            cam_map = np.array(cam_map, dtype=np.float32)
            # 将NaN值替换为0
            cam_map[np.isnan(cam_map)] = 0
            max_value = np.max(cam_map)
            if max_value == 0:
                max_value = 1e-7
            cam_map /= max_value
            sam_mask = np.array(sam_masks[img_id], dtype=np.float32)
            sam_mask[np.isnan(sam_mask)] = 0

            cam_flat = cam_map.flatten()
            sam_flat = sam_mask.flatten()
            intersection = (cam_flat * sam_flat).sum()
            loss = 1 - ((2. * intersection + smooth) / (cam_flat.sum() + sam_flat.sum() + smooth))
            total_loss_per_image += loss

        return total_loss_per_image / num_images if num_images != 0 else total_loss_per_image

    def clean_list(self):
        self.cam.clean_list()

    def forward(self, inputs, outputs, targets):
        batchsize = inputs.size()[0]
        losses = 0
        t0 = time.time()
        for i in range(batchsize):
            t1 = time.time()
            input_tensor = inputs[i]
            numpy_img = input_tensor.detach().cpu().numpy()
            rgb_numpy_img = numpy_img.transpose(1, 2, 0)
            rgb_numpy_img_uint8 = (255 * rgb_numpy_img).astype(np.uint8)
            # pil_image = Image.fromarray(rgb_numpy_img_uint8)
            # pil_image.show()
            targets_per_img = {}
            labels = targets[i]['labels'].tolist()
            boxes = targets[i]['boxes'].tolist()
            assert len(labels) == len(boxes), "difference in bboxes and labels"

            for label, box in zip(labels, boxes):
                if label not in targets_per_img:
                    targets_per_img[label] = []

                targets_per_img[label].append(box)
            t2 = time.time()
            cam_masks = self.cam_mask_generator(input_tensor, outputs["pred_logits"][i], targets_per_img)
            t3 = time.time()
            sam_masks = self.sam(rgb_numpy_img_uint8, targets_per_img)
            t4 = time.time()
            losses += self.loss_calculator(cam_maps=cam_masks, sam_masks=sam_masks)
            t5 = time.time()
            # print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)
        return losses / batchsize

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
