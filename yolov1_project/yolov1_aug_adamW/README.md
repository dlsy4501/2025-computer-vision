## YOLOv1 객체 탐지기 (ResNet-101 백본) - v2 업데이트

본 프로젝트는 **YOLOv1 (You Only Look Once v1)** 객체 탐지 네트워크를 기반으로, **ResNet-101**을 백본으로 채택하여 성능을 개선한 모델 구현입니다. 특히, **데이터 증강(Data Augmentation)** 기법을 강화하고 학습 설정을 일부 변경하여 모델의 **일반화 성능**을 더욱 향상시켰습니다.

-----

### 주요 개선점 및 업데이트 요약

| 특징 | 설명 |
| :--- | :--- |
| **랜덤 솔트 & 페퍼 노이즈 추가** | 훈련 데이터에 **랜덤 노이즈**를 적용하여 모델이 **변형 및 손상된 이미지**에도 강건하게 객체를 탐지하도록 **일반화 성능을 강화**했습니다.  |
| **ResNet-101 백본 사용** | 기존 YOLOv1보다 **깊은 피처 표현력**을 확보하여 복잡한 객체와 고해상도 이미지 처리 능력을 향상시킵니다. (ImageNet 사전학습 모델 활용) |
| **AdamW Optimizer 적용** | 가중치 감쇠(Weight Decay)를 효과적으로 관리하여 **안정적인 수렴**을 돕습니다. |
| **CosineAnnealing LR + Warm-up** | 학습률 스케줄링을 최적화하여 훈련 초기에 안정성을 높였습니다. |

-----

### 학습 환경 설정 (Configuration)

| 항목 | 설정값 | **변경 사항** |
| :--- | :--- | :--- |
| **백본 (Backbone)** | **ResNet-101** (pretrained on ImageNet) | - |
| **Epochs** | **70** | **150 $\rightarrow$ 70으로 변경** |
| **Batch Size** | 16 | - |
| **Optimizer** | **AdamW** | - |
| **Initial Learning Rate** | $1 \times 10^{-4}$ (0.0001) | - |
| **Weight Decay** | $1 \times 10^{-4}$ (0.0001) | - |
| **Scheduler** | CosineAnnealingLR | - |
| **Warm-up** | 10 epochs | - |
| **Input Size** | 512 x 512 | - |
| **Dataset** | VOC2012 (./Dataset/ 폴더) | - |
| **데이터 증강** | **랜덤 솔트 & 페퍼 노이즈 추가** | **신규 추가** |

-----

### 손실 함수 상세 (Loss Function Details)

딥 백본 환경에 맞게 가중치가 조정되었으며, 안정화를 위해 Smooth L1과 Cross Entropy를 사용합니다.

| 항목 | 설정값 | 설명 |
| :--- | :--- | :--- |
| $\lambda_{coord}$ | 3.0 | **위치 오차 가중** |
| $\lambda_{noobj}$ | 0.3 | **배경(No Object) 손실 감소** |
| $\lambda_{obj}$ | 1.0 | 객체 존재 확률 손실 |
| $\lambda_{cls}$ | 1.5 | **클래스 분류 손실 강화** |
| **손실 함수** | **Smooth L1 + CrossEntropy** | 박스 회귀 안정화 및 분류 성능 향상 |
| **IoU 계산 안정화** | $+1e^{-6}$ epsilon | 수치적 안정성을 위한 $\epsilon$ 추가 |

-----

### 데이터셋 구조

학습을 위한 데이터셋 구조는 VOC2012 형식으로 아래와 같습니다.

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

모델 학습을 시작합니다. Epochs가 70으로 변경되었음에 유의하십시오. 가중치는 `./weights/` 폴더에 주기적으로 저장됩니다.

```bash
python3 main.py
```

  * **최종 모델**: `yolov1_final.pth`

#### 2\. 모델 평가 (Evaluation)

학습된 모델의 정확도, Recall, **mAP** 등을 평가합니다.

```bash
python3 eval.py
```

  * **출력 항목**: Precision, Recall, mAP

#### 3\. 객체 탐지 (Detection)

새로운 이미지에서 객체 탐지를 수행합니다.

```bash
# 탐지 결과 화면 출력
python3 detect.py --image assets/person.jpg

# 탐지 결과 이미지 저장 (--save_img 옵션)
python3 detect.py --image assets/person.jpg --save_img
```

  * 결과 이미지는 `./output/` 디렉토리에 저장됩니다.

-----

### 프로젝트 실행 흐름

**\[Dataset 준비]** $\rightarrow$ **\[main.py 학습 진행]** $\rightarrow$ **\[eval.py 성능 평가]** $\rightarrow$ **\[detect.py 실 이미지 탐지]**
