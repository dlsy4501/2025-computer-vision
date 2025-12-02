import os
import cv2
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import sys

# ==========================================
# [1. 경로 자동 설정]
# detect.py가 있는 폴더(yolov1/yolov1)와 상위 폴더(yolov1)를 연결
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ==========================================

# 이제 nets와 utils를 정상적으로 불러올 수 있습니다.
try:
    from nets.yolo_resnet import YOLOv1_ResNet
    from utils.util import predict
except ImportError:
    print("Error: 모듈을 찾을 수 없습니다. (nets 또는 utils)")
    sys.exit()

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
          'tvmonitor': (128, 192, 0)
          }


def detect(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('LOADING MODEL...')
    # === [핵심 수정] resnet50() -> YOLOv1_ResNet() 교체 ===
    # 가중치 파일 구조(backbone...)와 일치하는 클래스를 사용해야 합니다.
    model = YOLOv1_ResNet(num_classes=20, S=7, B=2).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 가중치 로드
    weight_path = os.path.join(project_root, 'weights', args.weight)
    
    # 경로 안전장치: args로 입력받은 파일이 없으면 기본 경로 확인
    if not os.path.exists(weight_path):
         # detect.py가 있는 폴더 기준 weights 확인
         weight_path = os.path.join(current_file_dir, 'weights', args.weight)
    
    try:
        print(f"Loading weights from: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    except FileNotFoundError:
        print(f"Error: 가중치 파일을 찾을 수 없습니다 -> {weight_path}")
        return
    except RuntimeError as e:
        print(f"Error: 모델 구조 불일치. ({e})")
        return

    model.eval()
    
    with torch.no_grad():
        # 이미지 경로 설정 (Dataset/Images 폴더가 project_root에 있다고 가정)
        image_path = os.path.join(project_root, 'yolov1', 'assets', args.image)
        
        if not os.path.exists(image_path):
             print(f"Error: 이미지를 찾을 수 없습니다 -> {image_path}")
             return

        image = cv2.imread(image_path)
        print(f'\nPREDICTING {image_path}...')
        
        # util.py의 predict 함수 호출
        # (root_path 없이 전체 경로를 넘겨줍니다)
        result = predict(model, image_path)

    # 결과 시각화
    for x1y1, x2y2, class_name, _, prob in result:
        color = COLORS[class_name]
        cv2.rectangle(image, x1y1, x2y2, color, 2)

        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        p1 = (x1y1[0], x1y1[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    if args.save_img:
        save_path = os.path.join(project_root, 'result.jpg')
        cv2.imwrite(save_path, image)
        print(f"Result saved to {save_path}")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', default='person.jpg', required=False, help='Path to Image file')
    parser.add_argument('--save_img', action='store_true', help='Save the Image after detection')
    parser.add_argument('--video', default='', required=False, help='Path to Video file')  # maybe later
    parser.add_argument('--weight', default='yolov1_final.pth', required=False, help='Path to weight file') 
    args = parser.parse_args()

    detect(args)
