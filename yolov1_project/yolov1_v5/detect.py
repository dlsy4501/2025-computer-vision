import os
import cv2
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import sys


current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 상위 폴더 (프로젝트 루트): .../yolov1
project_root = os.path.dirname(current_file_dir)

# 모듈 찾기 위한 경로 추가 (nets, utils)
if current_file_dir not in sys.path:
    sys.path.append(current_file_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from nets.yolo_resnet import YOLOv1_ResNet
    from utils.util import predict
except ImportError:
    print("Error: 'nets' 또는 'utils' 모듈을 찾을 수 없습니다.")
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
    
    # [중요] 학습할 때 S=14로 바꿨다면 여기서도 S=14여야 합니다.
    # 만약 예전 S=7 가중치를 쓴다면 7로 변경하세요.
    model = YOLOv1_ResNet(num_classes=20, S=14, B=2).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ==========================================
    # [2] 가중치 파일 경로 찾기 로직
    # ==========================================
    weight_file = args.weight
    
    # 1. 현재 폴더/weights 확인 (방금 학습한 것이 여기 저장됨)
    path_local = os.path.join(current_file_dir, 'weights', weight_file)
    # 2. 프로젝트 루트/weights 확인 (옛날 가중치)
    path_root = os.path.join(project_root, 'weights', weight_file)
    
    if os.path.exists(path_local):
        weight_path = path_local
    elif os.path.exists(path_root):
        weight_path = path_root
    elif os.path.exists(weight_file): # 절대경로 입력 시
        weight_path = weight_file
    else:
        print(f"Error: 가중치 파일을 찾을 수 없습니다 -> {weight_file}")
        print(f"검색 위치 1: {path_local}")
        print(f"검색 위치 2: {path_root}")
        return

    # 가중치 로드
    try:
        print(f"Loading weights from: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError as e:
        print(f"\n[Error] 가중치 로드 실패: {e}")
        print("Tip: 학습된 모델의 S값(7 또는 14)과 현재 코드의 S값이 일치하는지 확인하세요.")
        return

    model.eval()

    # ==========================================
    # [3] 이미지 경로 찾기 로직
    # ==========================================
    image_name = args.image
    
    # 1. 현재 폴더/assets
    path_assets_local = os.path.join(current_file_dir, 'assets', image_name)
    # 2. 프로젝트 루트/yolov1/assets (이전 구조 호환)
    path_assets_root = os.path.join(project_root, 'yolov1', 'assets', image_name)
    # 3. Dataset/Images
    path_dataset = os.path.join(project_root, 'Dataset', 'Images', image_name)
    
    if os.path.exists(path_assets_local):
        final_image_path = path_assets_local
    elif os.path.exists(path_assets_root):
        final_image_path = path_assets_root
    elif os.path.exists(path_dataset):
        final_image_path = path_dataset
    elif os.path.exists(image_name): 
        final_image_path = image_name
    else:
        print(f"\n[Error] 이미지를 찾을 수 없습니다: {image_name}")
        return

    with torch.no_grad():
        image = cv2.imread(final_image_path)
        if image is None:
            print(f"[Error] 이미지를 읽을 수 없습니다. 경로: {final_image_path}")
            return

        print(f'\nPREDICTING: {final_image_path}...')
        
        # util.py의 predict 호출 (root_path 없이 전체 경로 전달)
        result = predict(model, final_image_path, root_path='')

    # 결과 그리기
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
        save_path = os.path.join(current_file_dir, 'result.jpg')
        cv2.imwrite(save_path, image)
        print(f"Result saved to: {save_path}")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', default='person.jpg', required=False, help='Image file name')
    parser.add_argument('--save_img', action='store_true', help='Save the Image after detection')
    parser.add_argument('--weight', default='yolov1_final.pth', required=False, help='Weight file name')
    args = parser.parse_args()

    detect(args)