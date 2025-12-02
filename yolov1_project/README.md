## YOLOv1 객체 탐지기 (ResNet-101 백본)

본 프로젝트는 **YOLOv1 (You Only Look Once v1)** 객체 탐지 네트워크를 기반으로, **ResNet-101**을 백본으로 채택하여 성능을 개선한 모델 구현입니다. 기존 YOLOv1의 얕은 네트워크 대신 ImageNet 사전 학습된 깊은 백본을 사용하여 **특징 표현력과 탐지 정확도**를 향상시켰습니다.

-----

### 주요 개선점 요약

| 특징 | 설명 |
| :--- | :--- |
| **ResNet-101 백본 사용** | 기존 YOLOv1보다 **깊은 피처 표현력**을 확보하여 복잡한 객체와 고해상도 이미지 처리 능력을 향상시킵니다. (ImageNet 사전학습 모델 활용) |
| **AdamW Optimizer 적용** | 가중치 감쇠(Weight Decay)를 효과적으로 관리하여 **안정적인 수렴** 및 **일반화 성능**을 개선합니다. |
| **CosineAnnealing LR + Warm-up** | 학습률 스케줄링을 최적화하여 훈련 초기에 안정성을 높이고 후반부에 **최적의 해**를 찾도록 돕습니다. |
| **손실 가중치 재조정** | 깊은 백본 환경에 맞게 $\lambda_{coord}$, $\lambda_{noobj}$, $\lambda_{cls}$ 가중치를 조정하여 **손실 함수의 균형**을 맞춥니다. |
| **Smooth L1 + CrossEntropy Loss** | 박스 회귀를 **Smooth L1**으로 안정화하고, 분류 정확도를 높였습니다. |

-----

### ⚙️ 학습 환경 설정 (Configuration)

| 항목 | 설정값 |
| :--- | :--- |
| **백본 (Backbone)** | **ResNet-101** (pretrained on ImageNet) |
| **Epochs** | 150 |
| **Batch Size** | 16 |
| **Optimizer** | **AdamW** |
| **Initial Learning Rate** | $1 \times 10^{-4}$ (0.0001) |
| **Weight Decay** | $1 \times 10^{-4}$ (0.0001) |
| **Scheduler** | CosineAnnealingLR |
| **Warm-up** | 10 epochs |
| **Input Size** | 512 x 512 |
| **Dataset** | VOC2012 (./Dataset/ 폴더) |

-----

### 손실 함수 상세 (Loss Function Details)

본 프로젝트에서는 YOLOv1의 기본 손실 함수를 바탕으로 딥 백본에 적합하게 가중치 및 안정성을 조정했습니다.

| 항목 | 설정값 | 설명 |
| :--- | :--- | :--- |
| $\lambda_{coord}$ | 3.0 | **위치 오차 가중** (Bounding Box Regression) |
| $\lambda_{noobj}$ | 0.3 | **배경(No Object) 손실 감소** |
| $\lambda_{obj}$ | 1.0 | 객체 존재 확률 손실 |
| $\lambda_{cls}$ | 1.5 | **클래스 분류 손실 강화** |
| **손실 함수** | **Smooth L1 + CrossEntropy** | 박스 회귀 안정화 및 분류 성능 향상 |
| **IoU 계산 안정화** | $+1e^{-6}$ epsilon | 수치적 안정성을 위한 $\epsilon$ 추가 |

-----

### 데이터셋 구조

학습을 위해 데이터셋은 아래와 같은 구조를 갖추어야 합니다.

```
./Dataset
├── Images
│   ├── 0001.jpg
│   ├── 0002.jpg
├── Labels
│   ├── 0001.txt
│   ├── 0002.txt
├── train.txt
└── test.txt
```

-----

### 🧑‍💻 프로젝트 실행 가이드

#### 1\. 훈련 (Training)

모델 학습을 시작합니다. 가중치는 `./weights/` 폴더에 주기적으로 저장됩니다.

```bash
python3 main.py
```

  * **저장 예시**: `yolov1_0010.pth`, `yolov1_final.pth`

#### 2\. 모델 평가 (Evaluation)

학습된 모델의 정확도, Recall, **mAP (mean Average Precision)** 등을 평가합니다.

```bash
python3 eval.py
```

  * **출력 항목**: Precision, Recall, mAP
  * 코드 내 `im_show=True` 설정 시 **탐지 결과 시각화**가 가능합니다.

#### 3\. 객체 탐지 (Detection)

새로운 이미지에 대해 객체 탐지를 수행하고 결과를 확인합니다.

```bash
# 탐지 결과 화면 출력
python3 detect.py --image assets/person.jpg

# 탐지 결과 이미지 저장 (--save_img 옵션)
python3 detect.py --image assets/person.jpg --save_img
```

  * 결과 이미지는 `./output/` 디렉토리에 저장됩니다.

-----

### 프로젝트 실행 흐름

**\[Dataset 준비]** $\rightarrow$ **\[main.py 학습 진행]** $\rightarrow$ **\[eval.py 성능 평가 (mAP, Precision)]** $\rightarrow$ **\[detect.py 실 이미지 탐지 및 시각화]**
