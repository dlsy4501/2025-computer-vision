import torch
import torch.nn as nn
import torchvision.models as models

class YOLOv1_ResNet(nn.Module):
    # 기본 S=14 설정
    def __init__(self, num_classes=20, S=14, B=2):
        super(YOLOv1_ResNet, self).__init__()

        self.S = S
        self.B = B
        self.C = num_classes
        
        # ResNet101 backbone
        # 448x448 입력 -> Layer4 통과 시 14x14 피처맵 (Stride 32)
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 

        # YOLO head
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, (self.B * 5 + self.C), kernel_size=1)
        )
        
    def forward(self, x):
        x = self.backbone(x)       # (N, 2048, 14, 14)
        x = self.conv(x)           # (N, 30, 14, 14)

        # 혹시 모를 크기 보정 (S=14라면 영향 없음)
        x = nn.functional.interpolate(x, size=(self.S, self.S), mode='bilinear')
        
        # 채널 차원 변경: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)  
        
        return x
