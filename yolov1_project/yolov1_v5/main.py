import os
import tqdm
import numpy as np
import random
import argparse
import re

import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet101 

# 작성된 모듈 임포트
from utils.loss import yoloLoss
from utils.dataset import Dataset
from nets.yolo_resnet import YOLOv1_ResNet

def main(args):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_file_dir, 'weights')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    warmup_epochs = args.warmup

    # Seed 설정
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ==========================================
    # [모델 초기화] Grid Cell = 14
    # ==========================================
    net = YOLOv1_ResNet(num_classes=20, S=14, B=2)

    if args.pre_weights is not None:
        print(f"Loading pretrained weights from {args.pre_weights}...")
        try:
            pattern = 'yolov1_([0-9]+)'
            strs = args.pre_weights.split('.')[-2]
            f_name = strs.split('/')[-1]
            if '\\' in f_name: 
                f_name = f_name.split('\\')[-1]
            epoch_str = re.search(pattern, f_name).group(1)
            epoch_start = int(epoch_str) + 1
        except:
            epoch_start = 1

        load_path = os.path.join(save_dir, args.pre_weights)
        if not os.path.exists(load_path):
             load_path = args.pre_weights # 절대 경로 시도
            
        state_dict = torch.load(load_path, map_location=device)['state_dict']
        net.load_state_dict(state_dict)
        print(f'Resuming from epoch {epoch_start}...')
    else:
        epoch_start = 1
        print('Loading ResNet101 ImageNet weights for backbone...')
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet_dict = resnet.state_dict()
        net_dict = net.state_dict()
        for k in resnet_dict.keys():
            if k in net_dict.keys() and not k.startswith('fc'):
                net_dict[k] = resnet_dict[k]
        net.load_state_dict(net_dict)

    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    # ==========================================
    # [손실 함수] Grid Cell = 14
    # ==========================================
    criterion = yoloLoss(S=14, B=2, num_class=20).to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Dataset 클래스 내부에서 색상 증강 등을 수행하므로 
    # 여기서는 Tensor 변환만 수행합니다.
    base_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    with open(os.path.join(root, 'train.txt')) as f:
        train_names = [line.strip() for line in f.readlines()]
    
    with open(os.path.join(root, 'test.txt')) as f:
        test_names = [line.strip() for line in f.readlines()]
    
    # S=14 명시적 전달
    train_dataset = Dataset(root, train_names, train=True, transform=[base_transform], S=14)
    test_dataset = Dataset(root, test_names, train=False, transform=[base_transform], S=14)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=os.cpu_count() if os.cpu_count() else 4, 
                                               pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False,
                                              num_workers=os.cpu_count() if os.cpu_count() else 4,
                                              pin_memory=True)

    print(f'NUMBER OF TRAIN DATA: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    def get_lr(epoch):
        if epoch < warmup_epochs:
            return learning_rate * (epoch + 1) / warmup_epochs
        else:
            return learning_rate

    for epoch in range(epoch_start, num_epochs + 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch)

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
            
            s = ('%10s' + '%10.4g' + '%10.4g' + '%10s') % (
                '%g/%g' % (epoch, num_epochs),
                optimizer.param_groups[0]['lr'],
                total_loss / (i + 1), 
                mem)
            progress_bar.set_description(s)

        scheduler.step()

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

            save = {'state_dict': net.state_dict()}
            torch.save(save, os.path.join(args.save_dir, f'yolov1_{epoch:04d}.pth'))

    save = {'state_dict': net.state_dict()}
    final_path = os.path.join(args.save_dir, 'yolov1_final.pth')
    torch.save(save, final_path)
    print(f'\nTraining Complete. Final weights saved to {final_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--data_dir", type=str, default='./Dataset', help="Dataset path")
    parser.add_argument("--pre_weights", type=str, default=None, help="Pretrained weights")
    parser.add_argument("--save_dir", type=str, default="./weights", help="Save directory")
    args = parser.parse_args()

    main(args)