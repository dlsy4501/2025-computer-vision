## YOLOv1 객체 탐지기 (ResNet-101 백본) - 성능 개선 버전

본 프로젝트는 **YOLOv1** 객체 탐지 모델에 **ResNet-101** 백본을 적용하여 피처 표현력을 강화하고, **데이터 증강 및 손실 함수 가중치, 그리드 셀($S$) 조정**을 통해 **작은 객체 탐지 성능(예: `pottedplant`, `diningtable`)을 집중적으로 개선**한 버전입니다.

-----

### 주요 업데이트 및 개선 사항

| 개선점 | 상세 내용 | 성능 영향 |
| :--- | :--- | :--- |
| **그리드 셀($\mathbf{S}$) 확장** | $7 \times 7$ 에서 \*\*$14 \times 14$\*\*로 변경. (B=3으로 증가) | 작은 객체에 대한 예측 민감도 향상 |
| **손실 함수 가중치 조정** | $\mathbf{\lambda_{coord}}$를 3.0 $\rightarrow$ **10.0**으로 상향.<br>$\mathbf{\lambda_{noobj}}$를 0.3 $\rightarrow$ **0.5**로 조정. | 바운딩 박스 위치 정확도 극대화 및 배경 오탐률 감소 |
| **색상 및 밝기 증강 추가** | `transforms.ColorJitter` 적용. | 다양한 조명 및 색상 조건에 대한 모델의 강건성 확보 |
| **백본 (Backbone)** | ResNet-101 (pretrained on ImageNet) 유지. | 깊은 특징 피처 추출 능력 유지 |
| **Optimizer & Scheduler** | AdamW 및 CosineAnnealingLR 유지. | 안정적인 수렴 및 최적의 학습률 스케줄링 |

-----

### 학습 환경 설정 (Configuration)

| 항목 | 설정값 |
| :--- | :--- |
| **백본 (Backbone)** | **ResNet-101** (pretrained on ImageNet) |
| **Epochs** | **70** |
| **Batch Size** | 16 |
| **Grid Size ($\mathbf{S}$)** | **14** |
| **Bounding Box ($\mathbf{B}$)** | **3** |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | $1 \times 10^{-4}$ (0.0001) |
| **Weight Decay** | $1 \times 10^{-4}$ (0.0001) |
| **Warm-up** | 10 epochs |
| **Input Size** | 512 x 512 |
| **Dataset** | VOC2012 (./Dataset/ 폴더) |

-----

### 손실 함수 상세 (Loss Function Details)

| 항목 | 설정값 | 설명 |
| :--- | :--- | :--- |
| $\mathbf{\lambda_{coord}}$ | **10.0** (기존 3.0) | **위치 오차 가중** |
| $\mathbf{\lambda_{noobj}}$ | **0.5** (기존 0.3) | **배경(No Object) 손실 감소 폭 조절** |
| $\mathbf{\lambda_{obj}}$ | 1.0 | 객체 존재 확률 손실 |
| $\mathbf{\lambda_{cls}}$ | 1.5 | 클래스 분류 손실 강화 |
| **손실 함수** | Smooth L1 + CrossEntropy | 박스 회귀 안정화 및 분류 성능 향상 |

-----

### 데이터셋 구조

```
./Dataset
├── Images
│   ├── 0001.jpg
├── Labels
│   ├── 0001.txt
├── train.txt
└── test.txt
```

-----

### 프로젝트 실행 가이드

#### 1\. 훈련 (Training)

학습을 진행합니다. 변경된 $\mathbf{S=14}$ 설정에 맞춰 `nets/yolo_resnet.py`와 `utils/loss.py` 파일의 구조가 수정되었는지 확인해야 합니다.

```bash
python3 main.py --epoch 70
```

  * **저장 경로**: `./weights/`

#### 2\. 모델 평가 (Evaluation)

학습된 모델의 mAP 등을 평가합니다. $\mathbf{mAP}$ 결과에서 `pottedplant`, `diningtable` 등의 개선 여부를 확인합니다.

```bash
python3 eval.py
```

  * **출력 항목**: Precision, Recall, mAP

#### 3\. 객체 탐지 (Detection)

새로운 이미지에서 객체 탐지를 수행합니다.

```bash
# 탐지 결과 화면 출력
python3 detect.py --image assets/sample.jpg

# 탐지 결과 이미지 저장
python3 detect.py --image assets/sample.jpg --save_img
```

  * 결과 이미지는 `./output/` 디렉토리에 저장됩니다.
