import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module

class yoloLoss(Module):
    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        # 🔹 수정된 하이퍼 파라미터
        self.lambda_coord = 3.0     # 좌표에 대한 가중 감소
        self.lambda_noobj = 0.3     # 배경 손실 축소
        self.lambda_obj = 1.0       # 객체 존재 여부 손실
        self.lambda_cls = 1.5       # 클래스 손실 비중 증가
        
        self.S = 14
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 14

        # 🔹 손실 함수 재정의
        self.smooth_l1 = torch.nn.SmoothL1Loss(reduction="sum")

    def compute_iou(self, box1, box2, index):
        box1 = torch.clone(box1)
        box2 = torch.clone(box2)
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
        inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))
        inter_h = torch.clamp(inter_h, min=0)
        inter_w = torch.clamp(inter_w, min=0)
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-6)  # 🔹 small epsilon for stability

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        box = torch.clamp(box, min=0)
        return box

    def forward(self, pred, target):
        batch_size = pred.size(0)

        # 🔹 [B, S, S, 30] → [B, S, S, 2, 5]
        target_boxes = target[:, :, :, :10].reshape(-1, self.S, self.S, 2, 5)
        pred_boxes = pred[:, :, :, :10].reshape(-1, self.S, self.S, 2, 5)
        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]

        obj_mask = (target_boxes[..., 4] > 0).bool()
        sig_mask = obj_mask[..., 1]  # [B, 14, 14]
        index = torch.where(sig_mask)

        for b, y, x in zip(*index):
            b, y, x = b.item(), y.item(), x.item()
            ious = self.compute_iou(pred_boxes[b, y, x, :, :4],
                                    target_boxes[b, y, x, :, :4],
                                    [x, y])
            _, max_i = ious.max(0)
            obj_mask[b, y, x, 1 - max_i] = 0

        noobj_mask = ~obj_mask

        # 🔸 no‑object confidence loss
        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][:, 4],
                                target_boxes[noobj_mask][:, 4],
                                reduction="sum")

        # 🔸 object confidence loss
        obj_loss = F.mse_loss(pred_boxes[obj_mask][:, 4],
                              target_boxes[obj_mask][:, 4],
                              reduction="sum")

        # 🔸 coordinate loss (SmoothL1)
        xy_loss = self.smooth_l1(pred_boxes[obj_mask][:, :2],
                                 target_boxes[obj_mask][:, :2])

        # 🔸 width/height 안정화 (sqrt 값 손실)
        wh_loss = self.smooth_l1(torch.sqrt(torch.clamp(pred_boxes[obj_mask][:, 2:4], min=1e-6)),
                                 torch.sqrt(torch.clamp(target_boxes[obj_mask][:, 2:4], min=1e-6)))

        # 🔸 class prediction loss
        class_loss = F.cross_entropy(pred_cls[sig_mask],
                                     target_cls[sig_mask].argmax(dim=1),
                                     reduction="sum")

        # 🔹 total loss (정규화 포함)
        total_loss = (
            self.lambda_coord * (xy_loss + wh_loss) +
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_cls * class_loss
        ) / batch_size

        return total_loss
