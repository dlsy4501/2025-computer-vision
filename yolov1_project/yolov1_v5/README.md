# YOLOv1-ResNet101 Improved

이 저장소는 PyTorch를 기반으로 **YOLOv1** 아키텍처를 구현한 프로젝트입니다. 백본(Backbone) 네트워크로 **ResNet101**을 사용하며, 원본 YOLOv1 대비 **작은 객체 탐지 성능**과 **조명 변화에 대한 강건함**을 개선하기 위해 다양한 최적화가 적용되었습니다.

##  주요 개선 사항 (Key Features)

이 구현체는 기존 YOLOv1의 한계를 극복하기 위해 다음과 같은 주요 변경 사항을 포함합니다:

1.  **고해상도 그리드 ($14 \times 14$ Grid):**

      * 기본 $7 \times 7$ 그리드를 \*\*$14 \times 14$\*\*로 확장하였습니다.
      * 이를 통해 입력 이미지(448x448)에서 더 세밀한 영역(32x32 픽셀 단위)을 분석하여, **작은 객체 탐지(Small Object Detection)** 성능을 크게 향상시켰습니다.

2.  **HSV 기반 데이터 증강 (Advanced Augmentation):**

      * 기존의 단순 밝기 조절을 넘어 **HSV(Hue, Saturation, Value)** 색 공간 기반의 증강 기법을 도입했습니다.
      * 다양한 색상 및 조명 환경에서도 객체를 잘 인식하도록 학습 파이프라인이 강화되었습니다.

3.  **손실 함수 및 학습 안정화:**

      * **가중치 조정:** 좌표 예측 손실($\lambda_{coord}=5.0$)과 배경(No Object) 손실($\lambda_{noobj}=0.5$)의 균형을 재조정했습니다.
      * **인코딩 버그 수정:** 학습 데이터 인코딩 과정에서 클래스 확률과 박스 신뢰도(Confidence) 인덱스가 겹치는 치명적인 버그를 수정하여 학습 정확도를 높였습니다.

4.  **ResNet101 Backbone:**

      * 강력한 특징 추출기인 ResNet101을 사용하여 복잡한 이미지에서도 높은 정확도의 Feature Map을 추출합니다.

-----

## 🛠️ 설치 및 환경 설정 (Requirements)

이 프로젝트는 Python 3.x 및 PyTorch 환경에서 동작합니다.

```bash
# 필요한 라이브러리 예시
pip install torch torchvision opencv-python tqdm matplotlib numpy
```

-----

##  프로젝트 구조 (Structure)

```bash
├── Dataset/
│   ├── Images/          # 학습 및 테스트 이미지 (.jpg)
│   ├── Labels/          # 정답 라벨 (.txt)
│   ├── train.txt        # 학습 이미지 파일명 리스트
│   └── test.txt         # 테스트 이미지 파일명 리스트
├── nets/
│   └── yolo_resnet.py   # YOLOv1 + ResNet101 모델 정의 (Grid 14)
├── utils/
│   ├── dataset.py       # 데이터셋 로더 및 HSV 증강 구현
│   ├── loss.py          # YOLO Loss 함수 (S=14, B=2)
│   └── util.py          # NMS 및 기타 유틸리티
├── weights/             # 학습된 가중치 저장소
├── main.py              # 학습 실행 스크립트
├── detect.py            # 객체 탐지 및 시각화 스크립트
└── eval.py              # mAP 성능 평가 스크립트
```

-----

##  사용 방법 (Usage)

### 1\. 데이터셋 준비 (Dataset Preparation)

데이터셋은 PASCAL VOC 형식을 따릅니다.

  * **Images:** 모든 이미지는 `Dataset/Images` 폴더에 위치해야 합니다.
  * **Labels:** 각 이미지에 대응하는 라벨 텍스트 파일은 `Dataset/Labels`에 위치해야 합니다.
      * 라벨 형식: `<class_idx> <x_center> <y_center> <width> <height>` (0\~1 정규화 된 값)
  * **List Files:** `Dataset/train.txt`와 `Dataset/test.txt`에 확장자를 제외한 파일명을 한 줄에 하나씩 기입합니다.

### 2\. 학습 (Training)

`main.py`를 실행하여 모델 학습을 시작합니다. 기본 배치 크기는 **16**으로 설정되어 있습니다.

```bash
# 기본 설정으로 학습 시작 (Grid 14, Batch 16)
python main.py

# 하이퍼파라미터 변경 예시
python main.py --epoch 100 --batch_size 16 --lr 0.0001 --save_dir ./weights
```

  * **GPU 메모리 주의:** ResNet101과 배치 크기 16 사용 시 VRAM이 부족할 경우, `--batch_size 8`로 낮추어 실행하세요.

### 3\. 추론 및 시각화 (Inference)

학습된 가중치를 사용하여 단일 이미지에 대한 객체 탐지를 수행합니다.

```bash
# 이미지 탐지 실행
python detect.py --image person.jpg --weight yolov1_final.pth

# 결과 이미지 저장 옵션 사용
python detect.py --image car.jpg --weight yolov1_final.pth --save_img
```

### 4\. 성능 평가 (Evaluation)

Test 셋에 대한 mAP(mean Average Precision)를 측정합니다.

```bash
python eval.py
```

-----

## 설정 (Configuration)

주요 하이퍼파라미터는 `main.py` 내부의 `argparse` 또는 코드를 통해 조정 가능합니다.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `S` (Grid Size) | **14** | 그리드 셀의 크기 (7 → 14로 상향) |
| `B` (Num Boxes) | 2 | 셀당 예측할 박스 개수 |
| `C` (Num Classes)| 20 | 클래스 개수 (PASCAL VOC 기준) |
| `batch_size` | **16** | 학습 배치 크기 |
| `epoch` | 70 | 전체 학습 에폭 수 |
| `lr` | 1e-4 | 학습률 (Learning Rate) |

-----

## Reference

  * Original Paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
  * Backbone: ResNet101 (Torchvision)
