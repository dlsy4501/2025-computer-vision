import os
import tqdm
import numpy as np
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms

from torchvision.models import resnet101
from utils.loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re

# ✅ 변경된 그리드 셀 및 바운딩 박스 설정
S = 14
B = 3

def main(args):
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    root = args.data_dir

    num_epochs = args.epoch
    batch_size = 16
    learning_rate = args.lr
    warmup_epochs = args.warmup

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ✅ S=14, B=3 설정 적용
    from nets.yolo_resnet import YOLOv1_ResNet
    net = YOLOv1_ResNet(num_classes=20, S=S, B=B)

    if args.pre_weights is not None:
        pattern = 'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        f_name = strs.split('/')[-1]
        epoch_str = re.search(pattern, f_name).group(1)
        epoch_start = int(epoch_str) + 1
        net.load_state_dict(torch.load(f'./weights/{args.pre_weights}')['state_dict'])
    else:
        epoch_start = 1
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet_dict = resnet.state_dict()
        
        net_dict = net.state_dict()
        for k in resnet_dict.keys():
            if k in net_dict.keys() and not k.startswith('fc'):
                net_dict[k] = resnet_dict[k]
        net.load_state_dict(net_dict)

    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    # ✅ 손실 함수 가중치 조정 (coord=10.0, noobj=0.5) 및 S, B 적용
    criterion = yoloLoss(S=S, B=B, num_class=20, lambda_coord=10.0, lambda_noobj=0.5).to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.train()

    # Optimizer 변경: AdamW (동일)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ✅ Dataset 준비 및 증강 추가
    # 훈련 데이터 증강: ColorJitter 추가
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    # ✅ Dataset에 수정된 train_transform 적용
    train_dataset = Dataset(root, train_names, train=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=os.cpu_count())

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    # ✅ Dataset에 수정된 test_transform 적용
    test_dataset = Dataset(root, test_names, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False,
                                                     num_workers=os.cpu_count())

    print(f'NUMBER OF TRAIN DATA: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')
    print(f'GRID SIZE (S): {S}, BOXES PER CELL (B): {B}')
    print(f'LAMBDA_COORD: 10.0, LAMBDA_NOOBJ: 0.5')
    
    # Warm-up 설정 (동일)
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return learning_rate * (epoch + 1) / warmup_epochs
        else:
            return learning_rate

    for epoch in range(epoch_start, num_epochs + 1):
        # Learning rate 설정
        current_lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        net.train()
        total_loss = 0.
        print(('\n' + '%10s' * 4) % ('epoch', 'lr', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, target) in progress_bar:
            images, target = images.to(device), target.to(device)

            pred = net(images)
            optimizer.zero_grad()
            loss = criterion(pred, target.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, num_epochs),
                                                             current_lr,
                                                             total_loss / (i + 1), mem)
            progress_bar.set_description(s)

        scheduler.step()

        # Validation loop (동일)
        if epoch % 5 == 0: 
            validation_loss = 0.0
            net.eval()
            with torch.no_grad():
                for i, (images, target) in enumerate(test_loader):
                    images, target = images.to(device), target.to(device)
                    prediction = net(images)
                    loss = criterion(prediction, target)
                    validation_loss += loss.item()

            validation_loss /= len(test_loader)
            print(f'Validation Loss: {validation_loss:07.3f}')
            
        # Save weights
        if epoch % 5 == 0:
            save = {'state_dict': net.state_dict()}
            torch.save(save, f'./weights/yolov1_{epoch:04d}.pth')

    # 최종 저장
    save = {'state_dict': net.state_dict()}
    torch.save(save, './weights/yolov1_final.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight path")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    # ✅ S와 B 인자를 yoloLoss에 전달하기 위해 main 함수 호출 시 인자를 수정해야 하지만,
    # yoloLoss가 main 함수 내에서 정의된 형태이므로,
    # 해당 부분은 yoloLoss와 YOLOv1_ResNet 정의부를 수정해야 합니다.
    # 여기서는 main 함수 상단에 S와 B를 전역 변수로 선언하고 이를 사용하도록 수정했습니다.
    # main(args) 

    # 파이썬 환경 실행을 위해 인자 값을 임시로 설정
    args.epoch = 70 # 이전 요청에서 70으로 설정 요청됨
    main(args)
