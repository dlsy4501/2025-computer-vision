import torch
import torch.nn.functional as F
from torch.nn import Module

class yoloLoss(Module):
    # S와 B를 인자로 받도록 수정
    def __init__(self, S=7, B=2, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = S  
        self.B = B  
        self.C = num_class
        self.step = 1.0 / self.S

    def compute_iou(self, box1, box2, index):
        box1 = torch.clone(box1)
        box2 = torch.clone(box2)
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
        inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))
        inter_w = torch.clamp(inter_w, 0)
        inter_h = torch.clamp(inter_h, 0)

        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter + 1e-6  # 0 나눗셈 방지
        return inter / union

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        return torch.clamp(box, 0, 1)

    def forward(self, pred, target):
        batch_size = pred.size(0)

        # bbox [batch, S, S, 2, 5]
        target_boxes = target[..., :10].contiguous().view(batch_size, self.S, self.S, self.B, 5)
        pred_boxes = pred[..., :10].contiguous().view(batch_size, self.S, self.S, self.B, 5)

        # class [batch, S, S, C]
        target_cls = target[..., 10:]
        pred_cls = pred[..., 10:]

        # obj mask
        obj_mask = (target_boxes[..., 4] > 0).bool()
        sig_mask = obj_mask.any(dim=-1)  # [batch, S, S], object 있는 cell

        # IOU 기반 bbox 선택
        index = torch.where(sig_mask)
        for img_i, y, x in zip(*index):
            img_i, y, x = int(img_i), int(y), int(x)
            pbox = pred_boxes[img_i, y, x]
            tbox = target_boxes[img_i, y, x]
            ious = self.compute_iou(pbox[:, :4], tbox[:, :4], [x, y])
            _, max_i = ious.max(0)
            obj_mask[img_i, y, x, 1 - max_i] = False

        noobj_mask = ~obj_mask

        # confidence loss
        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][..., 4],
                                target_boxes[noobj_mask][..., 4],
                                reduction="sum")
        obj_loss = F.mse_loss(pred_boxes[obj_mask][..., 4],
                              target_boxes[obj_mask][..., 4],
                              reduction="sum")

        # xy loss
        xy_loss = F.mse_loss(pred_boxes[obj_mask][..., :2],
                             target_boxes[obj_mask][..., :2],
                             reduction="sum")

        # wh loss (sqrt)
        pred_wh = pred_boxes[obj_mask][..., 2:4]
        pred_wh = torch.clamp(pred_wh, min=0)
        wh_loss = F.mse_loss(torch.sqrt(target_boxes[obj_mask][..., 2:4] + 1e-6),
                             torch.sqrt(pred_wh + 1e-6),
                             reduction="sum")

        # class loss
        sig_mask_exp = sig_mask.unsqueeze(-1).expand_as(pred_cls)
        class_loss = F.mse_loss(pred_cls[sig_mask_exp].view(-1, self.C),
                                target_cls[sig_mask_exp].view(-1, self.C),
                                reduction="sum")

        # total loss
        loss = obj_loss + self.lambda_noobj * noobj_loss + \
               self.lambda_coord * xy_loss + self.lambda_coord * wh_loss + \
               class_loss

        return loss / batch_size
