import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import torchvision.transforms as transforms
from torchvision.ops import nms

# ==========================================
# [1. 경로 자동 설정]
# 현재 파일 위치: .../yolov1/yolov1/utils/util.py
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트(inner): .../yolov1/yolov1
project_root = os.path.dirname(current_file_dir)
# 전체 루트(outer): .../yolov1
outer_root = os.path.dirname(project_root)

# 시스템 경로에 추가 (nets 폴더 찾기 위함)
if project_root not in sys.path:
    sys.path.append(project_root)
# ==========================================

try:
    # ResNet101 기반의 YOLOv1_ResNet 클래스 로드
    from nets.yolo_resnet import YOLOv1_ResNet
except ImportError:
    pass 

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = {'aeroplane': (0, 0, 0),
          'bicycle': (128, 0, 0),
          'bird': (0, 128, 0),
          'boat': (128, 128, 0),
          'bottle': (0, 0, 128),
          'bus': (128, 0, 128),
          'car': (0, 128, 128),
          'cat': (128, 128, 128),
          'chair': (64, 0, 0),
          'cow': (192, 0, 0),
          'diningtable': (64, 128, 0),
          'dog': (192, 128, 0),
          'horse': (64, 0, 128),
          'motorbike': (192, 0, 128),
          'person': (64, 128, 128),
          'pottedplant': (192, 128, 128),
          'sheep': (0, 64, 0),
          'sofa': (128, 64, 0),
          'train': (0, 192, 0),
          'tvmonitor': (128, 192, 0)}


def decoder(prediction):
    device = prediction.device
    grid_num = 7
    cell_size = 1. / grid_num
    
    boxes = []
    cls_indexes = []
    confidences = []
    
    # prediction shape: [7, 7, 30]
    prediction = prediction.data.squeeze() 
    
    contain1 = prediction[:, :, 4].unsqueeze(2)
    contain2 = prediction[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    
    mask1 = contain > 0.1
    mask2 = (contain == contain.max())
    mask = (mask1 + mask2).gt(0)
    
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = prediction[i, j, b * 5:b * 5 + 4]
                    contain_prob = prediction[i, j, b * 5 + 4]
                    
                    xy = torch.tensor([j, i], device=device, dtype=torch.float32) * cell_size
                    box[:2] = box[:2] * cell_size + xy
                    
                    box_xy = torch.FloatTensor(box.size()).to(device)
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    
                    max_prob, cls_index = torch.max(prediction[i, j, 10:], 0)
                    score = contain_prob * max_prob
                    
                    if score > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexes.append(cls_index)
                        confidences.append(score)

    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    boxes = torch.cat(boxes, 0)
    confidences = torch.tensor(confidences, device=device)
    cls_indexes = torch.stack(cls_indexes).to(device)
    
    keep = nms(boxes, confidences, iou_threshold=0.5)
    return boxes[keep], cls_indexes[keep], confidences[keep]


def predict(model, img_name, root_path=''):
    device = next(model.parameters()).device
    results = []
    
    if root_path:
        img_path = os.path.join(root_path, img_name)
    else:
        img_path = img_name

    img = cv2.imread(img_path)
    if img is None:
        return []

    h, w, _ = img.shape
    img_resized = cv2.resize(img, (448, 448))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    mean = np.array([123, 117, 104], dtype=np.float32)
    img_normalized = img_rgb - mean

    transform = transforms.Compose([transforms.ToTensor(), ])
    img_tensor = transform(img_normalized)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # === [수정된 부분] 차원 자동 감지 및 변환 ===
    prediction = model(img_tensor)

    # 모델이 (Batch, Channel, Height, Width)인 경우 -> (Batch, 30, 7, 7)
    if prediction.shape[1] == 30:
        prediction = prediction.permute(0, 2, 3, 1) # (N, H, W, C)로 변환
    
    # 모델이 이미 (Batch, Height, Width, Channel)인 경우 -> (Batch, 7, 7, 30)
    # 이때는 아무것도 하지 않고 넘어갑니다.
    # ==========================================
    
    boxes, cls_indexes, confidences = decoder(prediction)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cls_index = int(cls_indexes[i])
        conf = float(confidences[i])
        
        results.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], img_name, conf])
        
    return results


if __name__ == '__main__':
    # 1. Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('LOADING MODEL...')
    try:
        model = YOLOv1_ResNet(num_classes=20, S=7, B=2).to(device)
    except NameError:
        print("Error: YOLOv1_ResNet 클래스를 찾을 수 없습니다.")
        exit()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 2. Weights 로드 (경로 문제 해결)
    # 1순위: 바깥쪽 weights 폴더 확인 (outer_root/weights)
    weight_path = os.path.join(outer_root, 'weights', 'yolov1_final.pth')
    
    # 2순위: 안쪽 weights 폴더 확인 (project_root/weights) - 만약 구조가 다르다면
    if not os.path.exists(weight_path):
        weight_path = os.path.join(project_root, 'weights', 'yolov1_final.pth')

    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Weights loaded from {weight_path}")
    except FileNotFoundError:
        print(f"Error: 가중치 파일을 찾을 수 없습니다.")
        print(f"확인된 경로: {os.path.join(outer_root, 'weights', 'yolov1_final.pth')}")
        sys.exit()

    model.eval()
    
    # 3. 예측 테스트
    test_image_name = 'person.jpg' 
    # assets 폴더도 바깥쪽/안쪽 모두 확인
    test_root_path = os.path.join(outer_root, 'assets')
    if not os.path.exists(test_root_path):
         test_root_path = os.path.join(project_root, 'assets')

    full_path = os.path.join(test_root_path, test_image_name)
    
    if os.path.exists(full_path):
        print(f'\nPREDICTING {full_path}...')
        with torch.no_grad():
            # 테스트 시에는 root_path를 명시적으로 넘김
            result = predict(model, test_image_name, root_path=test_root_path)

        # 4. 시각화
        img = cv2.imread(full_path)
        if img is not None:
            for (x1, y1), (x2, y2), class_name, _, prob in result:
                color = COLORS[class_name]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {prob:.2f}"
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (x1, y1 - text_size[1])
                cv2.rectangle(img, (p1[0] - 1, p1[1] - baseline - 1), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(img, label, (p1[0], p1[1] + text_size[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

            save_path = os.path.join(test_root_path, 'result.jpg')
            cv2.imwrite(save_path, img)
            print(f"Result saved to {save_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.show()
    else:
        print(f"Test image not found at {full_path}")
