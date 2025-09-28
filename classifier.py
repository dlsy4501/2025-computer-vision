import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import numpy as np
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 데이터 증강을 적용하지 않은 Transform
transform_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 평균, 표준편차
])

# 2. 데이터 증강을 적용한 Transform
transform_augmented = transforms.Compose([
    transforms.RandomRotation(10), # 이미지를 -10도에서 10도 사이로 랜덤하게 회전
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 좌우, 상하 10% 이내로 랜덤 이동
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 전체 학습 데이터셋 로드
full_train_dataset_basic = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_basic)
full_train_dataset_augmented = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_augmented)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_basic)

# Train / Validation 분할 (55000 / 5000)
train_size = 55000
val_size = len(full_train_dataset_basic) - train_size
train_dataset_basic, val_dataset_basic = random_split(full_train_dataset_basic, [train_size, val_size])
train_dataset_augmented, val_dataset_augmented = random_split(full_train_dataset_augmented, [train_size, val_size])

# 데이터 로더 생성
train_loader_basic = DataLoader(train_dataset_basic, batch_size=BATCH_SIZE, shuffle=True)
val_loader_basic = DataLoader(val_dataset_basic, batch_size=BATCH_SIZE, shuffle=False)
train_loader_augmented = DataLoader(train_dataset_augmented, batch_size=BATCH_SIZE, shuffle=True)
# val_loader_augmented는 기본 transform을 사용해도 무방합니다. (검증 시에는 증강을 하지 않음)
val_loader_augmented = DataLoader(val_dataset_basic, batch_size=BATCH_SIZE, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)




#모델 구현
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: (batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # after pool: (batch, 32, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # after pool: (batch, 64, 7, 7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_resnet18_for_mnist():
    # 사전 학습된 ResNet18 모델 로드
    resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    # 1. 백본 가중치 동결 (Freeze)
    for param in resnet18.parameters():
        param.requires_grad = False

    # 2. 입력 채널 수정 (RGB 3채널 -> Grayscale 1채널)
    # 기존 가중치의 평균을 내어 1채널 가중치로 사용
    original_weights = resnet18.conv1.weight.clone()
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        resnet18.conv1.weight[:,:] = original_weights.mean(dim=1, keepdim=True)

    # 3. 출력층 수정 (1000개 클래스 -> 10개 클래스)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)
    
    # 새로 추가한 fc층과 수정한 conv1층의 파라미터만 학습되도록 설정
    for param in resnet18.conv1.parameters():
        param.requires_grad = True
    for param in resnet18.fc.parameters():
        param.requires_grad = True
        
    return resnet18



# 학습 및 평가 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train() # 훈련 모드
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # 검증
        model.eval() # 평가 모드
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training finished in {training_time:.2f} seconds")
    
    history = {'train_loss': train_loss_history, 'train_acc': train_acc_history,
               'val_loss': val_loss_history, 'val_acc': val_acc_history}
               
    return model, history, training_time

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 파라미터 수 계산 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 실험 결과 수집
# --- 실험 1: CNN (증강 X) ---
print("--- Starting Experiment 1: SimpleCNN without Augmentation ---")
cnn_basic = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_basic.parameters(), lr=LEARNING_RATE)
cnn_basic_model, cnn_basic_history, cnn_basic_time = train_model(cnn_basic, train_loader_basic, val_loader_basic, criterion, optimizer, EPOCHS)
cnn_basic_accuracy = test_model(cnn_basic_model, test_loader)
cnn_basic_params = count_parameters(cnn_basic_model)

# --- 실험 2: CNN (증강 O) ---
print("\n--- Starting Experiment 2: SimpleCNN with Augmentation ---")
cnn_aug = SimpleCNN().to(device)
optimizer = optim.Adam(cnn_aug.parameters(), lr=LEARNING_RATE)
cnn_aug_model, cnn_aug_history, cnn_aug_time = train_model(cnn_aug, train_loader_augmented, val_loader_augmented, criterion, optimizer, EPOCHS)
cnn_aug_accuracy = test_model(cnn_aug_model, test_loader)
cnn_aug_params = count_parameters(cnn_aug_model)

# --- 실험 3: ResNet18 (증강 X) ---
print("\n--- Starting Experiment 3: ResNet18 without Augmentation ---")
resnet_basic = get_resnet18_for_mnist().to(device)
# ResNet은 새로 추가된 레이어만 학습하므로, 해당 파라미터만 optimizer에 전달
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_basic.parameters()), lr=LEARNING_RATE)
resnet_basic_model, resnet_basic_history, resnet_basic_time = train_model(resnet_basic, train_loader_basic, val_loader_basic, criterion, optimizer, EPOCHS)
resnet_basic_accuracy = test_model(resnet_basic_model, test_loader)
resnet_basic_params = count_parameters(resnet_basic_model)

# --- 실험 4: ResNet18 (증강 O) ---
print("\n--- Starting Experiment 4: ResNet18 with Augmentation ---")
resnet_aug = get_resnet18_for_mnist().to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_aug.parameters()), lr=LEARNING_RATE)
resnet_aug_model, resnet_aug_history, resnet_aug_time = train_model(resnet_aug, train_loader_augmented, val_loader_augmented, criterion, optimizer, EPOCHS)
resnet_aug_accuracy = test_model(resnet_aug_model, test_loader)
resnet_aug_params = count_parameters(resnet_aug_model)



# 결과 시각화
# 학습 곡선 그리기 함수
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    plt.show()

# plot_history(cnn_basic_history, "CNN without Augmentation")
# ... 각 history에 대해 그래프 그리기

# 테스트셋 샘플 10개 추론
def inference_samples(model, data_loader, num_samples=10):
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images[:num_samples], labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 결과 출력 (보고서용 표 데이터)
    print("Image Index | True Label | Predicted Label")
    print("-----------------------------------------")
    for i in range(num_samples):
        print(f"{i+1:^11} | {labels[i]:^10} | {predicted[i]:^15}")
        # 이미지 시각화 (선택 사항)
        # plt.imshow(np.squeeze(images[i]), cmap='gray')
        # plt.title(f"True: {labels[i]}, Predicted: {predicted[i]}")
        # plt.show()

# 가장 성능이 좋았던 모델로 추론 수행
print("\n--- Inference on Test Samples (ResNet18 with Augmentation) ---")
inference_samples(resnet_aug_model, test_loader)
