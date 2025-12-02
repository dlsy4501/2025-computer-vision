import torch
import torch.nn as nn
import torchvision.models as models

class YOLOv1_ResNet(nn.Module):
    def __init__(self, num_classes=20, S=7, B=2):
        super(YOLOv1_ResNet, self).__init__()

        self.S = S
        self.B = B
        self.C = num_classes
        
        # ResNet101 backbone
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool, fc

        # YOLO head
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, (self.B * 5 + self.C), kernel_size=1)
        )
        
    def forward(self, x):
        x = self.backbone(x)       # (N, 2048, 16, 16)
        x = self.conv(x)           # (N, 30, 16, 16)

        # Resize to YOLO format (N, S, S, 30)
        x = nn.functional.interpolate(x, size=(self.S, self.S), mode='bilinear')
        x = x.permute(0, 2, 3, 1)
        return x
