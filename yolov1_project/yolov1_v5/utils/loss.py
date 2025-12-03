import torch
import torch.nn as nn
import torch.nn.functional as F

class yoloLoss(nn.Module):
    def __init__(self, S=14, B=2, num_class=20):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = num_class
        
        # [요청하신 손실함수 가중치 조정]
        # 작은 객체 탐지 성능을 위해 좌표 손실(coord)은 높게 유지
        # 배경 오탐지 방지를 위해 noobj 손실은 0.5로 설정
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def compute_iou(self, box1, box2):
        '''
        box1: (2, 4) -> [x, y, w, h] (prediction)
        box2: (2, 4) -> [x, y, w, h] (target)
        '''
        # 중앙점(x, y)와 너비/높이(w, h)를 좌상단(x1, y1), 우하단(x2, y2) 좌표로 변환
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

        # 교차 영역(Intersection) 계산
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

        # 합집합 영역(Union) 계산
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
        return iou

    def forward(self, pred, target):
        # pred: [batch, 14, 14, 30]
        # target: [batch, 14, 14, 30]
        
        batch_size = pred.size(0)
        
        # [데이터셋 수정 반영]
        # Box 정보: 0~9 (5 * 2)
        # Class 정보: 10~29 (20)
        target_boxes = target[..., :10].contiguous().view(batch_size, self.S, self.S, self.B, 5)
        pred_boxes = pred[..., :10].contiguous().view(batch_size, self.S, self.S, self.B, 5)
        
        target_cls = target[..., 10:]
        pred_cls = pred[..., 10:]

        # --- 1. Mask 생성 ---
        # 객체가 존재하는 셀 확인 (Confidence > 0)
        # Dataset에서 이미 Box1, Box2 Conf를 1로 설정했으므로 하나만 확인해도 됨
        obj_mask = target_boxes[..., 4] > 0 
        has_obj_mask = obj_mask[..., 0] # [batch, S, S]
        noobj_mask = ~obj_mask # 물체가 없는 곳

        # --- 2. No Object Loss ---
        # 물체가 없는 셀의 Confidence 점수를 0으로 학습
        noobj_loss = F.mse_loss(
            pred_boxes[..., 4][noobj_mask], 
            target_boxes[..., 4][noobj_mask], 
            reduction='sum'
        )

        # --- 3. Object Loss & Coordinate Loss & Class Loss ---
        obj_loss = 0
        coord_loss = 0
        class_loss = 0
        
        # 물체가 있는 셀들에 대해서만 루프
        batch_idx, grid_y, grid_x = torch.where(has_obj_mask)

        for b, y, x in zip(batch_idx, grid_y, grid_x):
            # 예측 박스 2개
            p_boxes = pred_boxes[b, y, x] # [2, 5]
            # 정답 박스 (dataset.py에서 동일하게 넣어줌)
            t_box = target_boxes[b, y, x] # [2, 5]
            
            # IoU 계산하여 책임 박스(Responsible Box) 선정
            ious = self.compute_iou(p_boxes[:, :4], t_box[:, :4])
            best_iou, best_idx = torch.max(ious, 0)
            
            # Best Box (책임 박스)
            p_box_best = p_boxes[best_idx]
            t_box_best = t_box[best_idx]

            # (1) Coordinate Loss
            # x, y 손실
            xy_loss = F.mse_loss(p_box_best[:2], t_box_best[:2], reduction='sum')
            
            # w, h 손실 (sqrt 적용)
            p_wh = torch.sqrt(torch.abs(p_box_best[2:4]) + 1e-6)
            t_wh = torch.sqrt(t_box_best[2:4])
            wh_loss = F.mse_loss(p_wh, t_wh, reduction='sum')
            
            coord_loss += (xy_loss + wh_loss)

            # (2) Object Confidence Loss
            obj_loss += F.mse_loss(p_box_best[4], t_box_best[4], reduction='sum')
            
            # (3) Non-Responsible Box Penalty
            # 물체가 있는 셀이지만, IoU가 낮아 선택받지 못한 박스도 No Object로 간주
            non_best_idx = 1 - best_idx
            noobj_loss += self.lambda_noobj * F.mse_loss(
                p_boxes[non_best_idx, 4], 
                torch.tensor(0.0).to(pred.device), 
                reduction='sum'
            )

        # (4) Class Loss
        if len(batch_idx) > 0:
            class_loss = F.mse_loss(
                pred_cls[has_obj_mask], 
                target_cls[has_obj_mask], 
                reduction='sum'
            )

        # 최종 Loss 합산
        total_loss = (self.lambda_coord * coord_loss) + obj_loss + (self.lambda_noobj * noobj_loss) + class_loss
        
        return total_loss / batch_size
