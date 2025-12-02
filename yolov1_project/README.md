📘 YOLOv1 객체 탐지기 (ResNet101 백본)
📄 프로젝트 개요
본 프로젝트는 YOLOv1 (You Only Look Once v1) 구조를 기반으로
ResNet‑101 백본을 사용한 객체 탐지 모델을 구현한 것입니다.

딥러닝 네트워크의 표현력을 향상시키기 위해 기존 얕은 YOLOv1 백본 대신
ImageNet 사전학습 모델(ResNet101) 을 사용하여 성능을 개선하였습니다.

⚙️ 학습 환경 설정
항목	설정값
백본(Backbone)	ResNet‑101 (pretrained on ImageNet)
Epochs	150
Batch Size	16
Optimizer	AdamW
Initial Learning Rate	1 × 10⁻⁴ (0.0001)
Weight Decay	1 × 10⁻⁴ (0.0001)
Scheduler	CosineAnnealingLR
Warm‑up	10 epochs
Input Size	512 × 512
Dataset	VOC2012 (./Dataset/ 폴더)
📦 데이터셋 구조
./Dataset ├── Images │ ├── 0001.jpg │ ├── 0002.jpg ├── Labels │ ├── 0001.txt │ ├── 0002.txt ├── train.txt └── test.txt
🧩 손실 함수 (Loss Function)
본 프로젝트에서는 YOLOv1 기본 손실식을 바탕으로
딥백본에 적합하도록 가중치 및 안정성을 조정하였습니다.

항목	설정값	설명
λ_coord	3.0	위치 오차 가중
λ_noobj	0.3	배경 오브젝트 손실 감소
λ_obj	1.0	객체 존재 확률 손실
λ_cls	1.5	클래스 분류 손실 강화
손실 함수	Smooth L1 + CrossEntropy	박스 회귀 안정화 및 분류 향상
IoU 계산	+1e‑6 epsilon	수치 안정화
🚀 학습 실행 방법
bash

python3 main.py
학습이 진행되면 ./weights/ 폴더에 주기적으로 가중치가 저장됩니다.
예: yolov1_0010.pth, yolov1_0020.pth
최종 모델: yolov1_final.pth
🧪 모델 평가 (Evaluation)
학습된 모델의 정확도 및 mAP 등을 평가합니다.

bash

python3 eval.py
출력: Precision, Recall, mAP
코드 내 im_show=True 설정 시 탐지 결과 시각화 가능
🔍 객체 탐지 (Detection)
새로운 이미지에서 객체 탐지를 수행합니다.

bash

# 탐지 결과 화면 출력
python3 detect.py --image assets/person.jpg

# 탐지 결과 이미지 저장
python3 detect.py --image assets/person.jpg --save_img
결과 이미지는 ./output/ 디렉토리에 저장됩니다.
📈 주요 개선점 요약
✅ ResNet101 백본 사용 → 피처 표현력 향상
✅ AdamW Optimizer 적용 → 안정적인 수렴 및 일반화
✅ CosineAnnealing LR + Warm‑up → 학습률 스케줄 최적화
✅ 손실 가중치 재조정 → 딥백본에 맞는 균형 유지
✅ 데이터 증강 강화 → 일반화 성능 향상
📊 프로젝트 실행 흐름
[Dataset 준비] ↓ [main.py 학습 진행] ↓ [eval.py 성능 평가 (mAP, Precision)] ↓ [detect.py 실 이미지 탐지 및 시각화]
🧠 요약
본 프로젝트는 YOLOv1 객체 탐지기를 ResNet‑101 백본과 AdamW 최적화 조합으로
재구성하여,
고해상도 입력과 깊은 네트워크 환경에서도 보다 안정적이고 향상된 성능을 달성하도록 설계되었습니다.
