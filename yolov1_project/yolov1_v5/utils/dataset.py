import os
import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2

class Dataset(data.Dataset):
    image_size = 448
    
    # Grid Num을 14로 상향 조정 (작은 객체 탐지 성능 향상)
    def __init__(self, root, file_names, train, transform, S=14, B=2, C=20):
        print('DATA INITIALIZATION')
        
        self.root_images = os.path.join(root, 'Images')
        self.root_labels = os.path.join(root, 'Labels')
        self.train = train
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
        self.f_names = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB

        for line in file_names:
            line = line.rstrip()
            with open(f"{self.root_labels}/{line}.txt") as f:
                objects = f.readlines()
                self.f_names.append(line + '.jpg')
                box = []
                label = []
                for object in objects:
                    c, x1, y1, x2, y2 = map(float, object.rstrip().split())
                    box.append([x1, y1, x2, y2])
                    # 파일의 class index(0~19)를 그대로 사용
                    label.append(int(c))
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        f_name = self.f_names[idx]
        img_path = os.path.join(self.root_images, f_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Image not found {img_path}")
            # 에러 방지를 위해 더미 이미지 생성 혹은 예외 처리
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            # 데이터 증강 파이프라인 (색상/밝기 증강 포함)
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            
            # [요청하신 색상 및 밝기 증강] HSV 기반으로 통합 적용
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)
        
        h, w, _ = img.shape
        # 박스 정규화 (0~1)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        
        img = self.BGR2RGB(img)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Grid 14 적용된 인코더 호출
        target = self.encoder(boxes, labels) 
        
        # PyTorch Tensor 변환
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        # Grid size S=14
        grid_num = self.S
        target = torch.zeros((grid_num, grid_num, self.B * 5 + self.C))
        cell_size = 1. / grid_num
        
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            # Grid cell 좌표 (ij)
            ij = (cxcy_sample / cell_size).ceil() - 1
            
            # 인덱스 범위 체크 (안전장치)
            grid_x = int(ij[0].clamp(0, grid_num-1))
            grid_y = int(ij[1].clamp(0, grid_num-1))
            
            # [수정됨] 채널 구조: [x,y,w,h,conf] * 2 + [Classes...]
            # Box 1 Confidence
            target[grid_y, grid_x, 4] = 1
            # Box 2 Confidence
            target[grid_y, grid_x, 9] = 1
            
            # [버그 수정] Class Probability
            # 기존 코드: int(labels[i]) + 9 -> Class 0인 경우 Index 9가 되어 Box2 Conf와 덮어씌워짐
            # 수정 코드: int(labels[i]) + 10 -> Box 정보(0~9) 뒤에 위치
            class_idx = int(labels[i]) + 10
            target[grid_y, grid_x, class_idx] = 1
            
            # Box 좌표 오프셋 및 크기 저장
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            
            # Box 1 (0~4)
            target[grid_y, grid_x, :2] = delta_xy
            target[grid_y, grid_x, 2:4] = wh[i]
            
            # Box 2 (5~9)
            target[grid_y, grid_x, 5:7] = delta_xy
            target[grid_y, grid_x, 7:9] = wh[i]
            
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v.astype(float) * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s.astype(float) * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.8, 1.2]) # Hue는 너무 많이 바꾸면 클래스가 달라질 수 있어 범위 축소
            h = h.astype(float) * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x), :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x), :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes