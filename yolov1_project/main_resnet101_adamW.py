import os
import tqdm
import numpy as np
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet101     # ✅ ResNet50 → ResNet101로 교체
from utils.loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    warmup_epochs = args.warmup

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = resnet101()

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

    criterion = yoloLoss().to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.train()

    # ✅ Optimizer 변경: AdamW
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ✅ Dataset 준비
    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=os.cpu_count())

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False,
                                              num_workers=os.cpu_count())

    print(f'NUMBER OF TRAIN DATA: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    # ✅ Warm-up 설정
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return learning_rate * (epoch + 1) / warmup_epochs
        else:
            return learning_rate

    for epoch in range(epoch_start, num_epochs + 1):
        # Learning rate 설정
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
            s = ('%10s' + '%10.4g' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, num_epochs),
                                                           optimizer.param_groups[0]['lr'],
                                                           total_loss / (i + 1), mem)
            progress_bar.set_description(s)

        scheduler.step()

        # ✅ Validation loop
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
        if epoch % 10 == 0:
            save = {'state_dict': net.state_dict()}
            torch.save(save, f'./weights/yolov1_{epoch:04d}.pth')

    # ✅ 최종 저장
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

    main(args)
