# 2025년 성신여자대학교 AI융합학부 여름방학 비교과 프로그램

## 딥러닝 서버 실습 및 파이토치 튜토리얼

### 프로그램 개요

**목적**

- 딥러닝 서버 실습을 통한 인공지능 전문가의 기초 실무 능력을 함양
- 심층학습 교과목 이론 모델을 구현하며 전공 교과목에 대한 이해를 심화
- 다양한 딥러닝 모델을 만들기 위한 준비와 기초 코딩 능력의 향상

**대상**

- 딥러닝 서버를 처음 사용하는 학생들
- Windows 또는 Mac 노트북 사용자

**교육 기간**

- 총 4일(8시간) 교육
- 일정: 2025년 8월 18일 ~ 8월 22일

### 서버 정보

**딥러닝 서버 접속 정보**

- IP 주소: `210.125.91.90`
- 포트: `22`
- 사용자명: `y*******`
- 비밀번호: `************`
- 사용 가능한 GPU: 0번 ~ 7번 (총 8개)

### 강의 일정

| 차시 | 날짜        | 시간        | 강의자 | 내용                                                          |
| ---- | ----------- | ----------- | ------ | ------------------------------------------------------------- |
| 01차 | 8월18일(월) | 10:00~12:00 | 고원준 | 딥러닝 서버 활용 방법과 파이토치 신경망 모델 기본 구성 만들기 |
| 02차 | 8월19일(화) | 10:00~12:00 | 고원준 | CNN 모델과 객체 인식 모델의 기초와 예제 익히기                |
| 03차 | 8월21일(목) | 10:00~12:00 | 안양준 | DCGAN과 시계열 모델 기초와 예제 익히기                        |
| 04차 | 8월22일(금) | 10:00~12:00 | 안양준 | NLP 모델 기초와 예제 익히기                                   |

---

## 사전 준비사항

### 1. 필요한 소프트웨어 설치

#### Windows 사용자

1. **PuTTY** 또는 **Windows Terminal** 설치
2. **Visual Studio Code** 설치
3. **Git** 설치 (선택사항)

#### Mac 사용자

1. **Terminal** (기본 설치됨)
2. **Visual Studio Code** 설치
3. **Git** 설치 (선택사항)

### 2. Visual Studio Code 확장 프로그램 설치

- Remote - SSH
- Python
- Jupyter

---

## 01차: 딥러닝 서버 활용 방법과 파이토치 신경망 모델 기본 구성 만들기

### 강의자: 고원준 | 8월18일(월) 10:00~12:00

### 1. 딥러닝 서버 접속 방법

#### SSH 클라이언트를 이용한 접속

**Windows 사용자 (PuTTY)**

1. PuTTY 실행
2. Host Name: `210.125.91.90`
3. Port: `22`
4. Connection type: SSH
5. Open 클릭
6. 사용자명: `yangjunahn`
7. 비밀번호: `0000`

**Mac/Linux 사용자**

```bash
ssh yangjunahn@210.125.91.90
```

#### Visual Studio Code를 이용한 접속

1. VSCode 실행
2. `Ctrl+Shift+P` (또는 `Cmd+Shift+P`) - Command Palette 열기
3. `Remote-SSH: Connect to Host` 선택
4. `yangjunahn@210.125.91.90` 입력
5. 비밀번호 `0000` 입력

### 2. 서버 환경 설정

#### 작업 디렉토리 생성

```bash
mkdir -p ~/pytorch_tutorial
cd ~/pytorch_tutorial
```

#### Python 가상환경 설정

```bash
python3 -m venv venv
source venv/bin/activate
```

#### PyTorch 설치 확인

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. GPU 할당 및 사용 방법

#### GPU 상태 확인

```bash
nvidia-smi
```

#### GPU 할당 방법

학생들은 다음과 같이 GPU를 할당받아 사용합니다:

- 학생 1-2: GPU 0번
- 학생 3-4: GPU 1번
- 학생 5-6: GPU 2번
- 학생 7-8: GPU 3번
- 학생 9-10: GPU 4번
- 학생 11-12: GPU 5번
- 학생 13-14: GPU 6번
- 학생 15-16: GPU 7번

#### 코드에서 GPU 지정

```python
import torch
import os

# GPU 지정 (예: 0번 GPU 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")
```

### 4. 파이토치 기본 익히기

#### 4.1 텐서(Tensor) 기초

**실습 파일 생성: `tensor_basics.py`**

```python
import torch
import numpy as np

# 텐서 생성
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"텐서 생성: {x_data}")

# NumPy 배열로부터 텐서 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"NumPy에서 텐서 생성: {x_np}")

# 다른 텐서로부터 텐서 생성
x_ones = torch.ones_like(x_data)
print(f"ones_like: {x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"rand_like: {x_rand}")

# 랜덤 또는 상수 값을 갖는 텐서 생성
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"랜덤 텐서: {rand_tensor}")
print(f"1로 채워진 텐서: {ones_tensor}")
print(f"0으로 채워진 텐서: {zeros_tensor}")
```

#### 4.2 신경망 모델 기본 구성

**실습 파일 생성: `neural_network_basics.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 모델 생성
model = NeuralNetwork()
print(model)

# 모델 파라미터 확인
print("모델 파라미터:")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")
```

#### 4.3 FashionMNIST 데이터셋 로드

**실습 파일 생성: `fashionmnist_loader.py`**

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 학습 데이터 다운로드
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 테스트 데이터 다운로드
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# 데이터로더 생성
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"X의 형태: {X.shape}")
    print(f"y의 형태: {y.shape} {y.dtype}")
    break

# 클래스 이름
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

print("클래스 목록:")
for key, value in labels_map.items():
    print(f"{key}: {value}")
```

### 5. 실습 과제

1. **텐서 조작 실습**: 주어진 텐서를 다양한 방법으로 조작해보기
2. **간단한 신경망 구성**: 입력층, 은닉층, 출력층을 갖는 신경망 구성하기
3. **데이터 로딩 실습**: FashionMNIST 데이터셋을 로드하고 탐색하기

### 6. 다음 차시 예고

다음 시간에는 CNN 모델의 기초와 이미지 분류를 위한 합성곱 신경망을 구현해보겠습니다.

---

## 02차: CNN 모델과 객체 인식 모델의 기초와 예제 익히기

### 강의자: 고원준 | 8월19일(화) 10:00~12:00

### 1. CNN 기초 이론

#### 합성곱 신경망(CNN) 개념

- 합성곱층(Convolution Layer)
- 풀링층(Pooling Layer)
- 완전연결층(Fully Connected Layer)

### 2. CNN 모델 구현

**실습 파일 생성: `cnn_model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 모델 생성 및 GPU 할당
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print(f"모델이 {device}에서 실행됩니다.")
print(model)
```

### 3. 이미지 분류 모델 학습

**실습 파일 생성: `train_cnn.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import CNN

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 데이터셋 로드
train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 모델, 손실함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
      
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
      
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
      
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
  
    accuracy = 100 * correct / total
    print(f'Training Accuracy: {accuracy:.2f}%')

# 테스트 함수
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
  
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
  
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 학습 실행
for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)
```

### 4. 전이학습(Transfer Learning) 실습

**실습 파일 생성: `transfer_learning.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 데이터 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 사전 훈련된 ResNet18 모델 로드
model = models.resnet18(pretrained=True)

# 분류기 부분만 교체
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # FashionMNIST 클래스 수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 옵티마이저 설정
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print("전이학습 모델 준비 완료!")
```

### 5. 객체 인식 모델 기초

**실습 파일 생성: `object_detection_intro.py`**

```python
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 사전 훈련된 객체 검출 모델 로드
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO 데이터셋 클래스 이름
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 객체 검출 함수
def detect_objects(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
  
    with torch.no_grad():
        predictions = model(image_tensor)
  
    return predictions, image

# 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
])

print("객체 검출 모델 준비 완료!")
print(f"검출 가능한 클래스 수: {len(COCO_INSTANCE_CATEGORY_NAMES)}")
```

### 6. 실습 과제

1. **CNN 모델 구현**: 다른 구조의 CNN 모델 구성해보기
2. **하이퍼파라미터 조정**: 학습률, 배치 크기 등을 변경하여 성능 비교
3. **전이학습 실습**: 다른 사전 훈련된 모델로 전이학습 수행

---

## 03차: DCGAN과 시계열 모델 기초와 예제 익히기

### 강의자: 안양준 | 8월21일(목) 10:00~12:00

### 1. DCGAN 기초 이론

#### GAN(Generative Adversarial Network) 개념

- 생성자(Generator)와 판별자(Discriminator)의 대립적 학습
- DCGAN (Deep Convolutional GAN)의 특징

### 2. DCGAN 구현

**실습 파일 생성: `dcgan_model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 생성자 네트워크
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# 판별자 네트워크
class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# 모델 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)

# 가중치 초기화
netG.apply(weights_init)
netD.apply(weights_init)

print("DCGAN 모델 생성 완료!")
```

### 3. DCGAN 학습

**실습 파일 생성: `train_dcgan.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dcgan_model import Generator, Discriminator

# 하이퍼파라미터
batch_size = 128
nz = 100  # 잠재 벡터 크기
lr = 0.0002
beta1 = 0.5
num_epochs = 5

# 데이터 로더 설정
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)

# 손실함수와 옵티마이저
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 학습 함수
def train_dcgan():
    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):
            # 판별자 학습
            netD.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
          
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
          
            # 가짜 데이터로 판별자 학습
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
          
            # 생성자 학습
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
          
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'Loss_D: {errD_real.item() + errD_fake.item():.4f} '
                      f'Loss_G: {errG.item():.4f}')

# 학습 시작
train_dcgan()
print("DCGAN 학습 완료!")
```

### 4. 시계열 모델 기초

#### LSTM을 이용한 시계열 예측

**실습 파일 생성: `lstm_model.py`**

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
      
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                           torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 데이터 생성 함수
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 샘플 데이터 생성 (사인파)
def generate_sine_wave(seq_length):
    x = np.linspace(0, 4 * np.pi, seq_length)
    y = np.sin(x)
    return y

# 모델 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("LSTM 모델 생성 완료!")
```

### 5. 시계열 데이터 학습

**실습 파일 생성: `train_lstm.py`**

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from lstm_model import LSTM, create_inout_sequences, generate_sine_wave

# 데이터 생성
seq_length = 1000
data = generate_sine_wave(seq_length)
train_data = data[:800]
test_data = data[800:]

# 데이터 정규화
train_data_normalized = (train_data - np.mean(train_data)) / np.std(train_data)
test_data_normalized = (test_data - np.mean(train_data)) / np.std(train_data)

# 시퀀스 생성
train_window = 12
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_lstm():
    epochs = 150
  
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
          
            y_pred = model(torch.FloatTensor(seq).to(device))
            single_loss = loss_function(y_pred, torch.FloatTensor(labels).to(device))
            single_loss.backward()
            optimizer.step()
      
        if i % 25 == 0:
            print(f'Epoch {i} loss: {single_loss.item()}')

# 예측 함수
def predict_future(model, test_inputs, future_steps):
    model.eval()
    predictions = []
  
    with torch.no_grad():
        for _ in range(future_steps):
            seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
          
            y_pred = model(seq)
            predictions.append(y_pred.item())
            test_inputs.append(y_pred.item())
  
    return predictions

# 학습 실행
train_lstm()

# 예측 실행
test_inputs = train_data_normalized[-train_window:].tolist()
predictions = predict_future(model, test_inputs, len(test_data_normalized))

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_data_normalized)), train_data_normalized, label='Training Data')
plt.plot(range(len(train_data_normalized), len(train_data_normalized) + len(test_data_normalized)), 
         test_data_normalized, label='Actual')
plt.plot(range(len(train_data_normalized), len(train_data_normalized) + len(predictions)), 
         predictions, label='Predicted')
plt.legend()
plt.title('LSTM Time Series Prediction')
plt.savefig('lstm_prediction.png')
plt.show()

print("LSTM 학습 및 예측 완료!")
```

### 6. 실습 과제

1. **DCGAN 실험**: 다른 하이퍼파라미터로 학습하고 결과 비교
2. **시계열 예측**: 다른 시계열 데이터(주식, 날씨 등)로 LSTM 모델 학습
3. **모델 개선**: RNN, GRU 등 다른 순환 신경망 구조 실험

---

## 04차: NLP 모델 기초와 예제 익히기

### 강의자: 안양준 | 8월22일(금) 10:00~12:00

### 1. NLP 기초 이론

#### 자연어 처리 개요

- 토큰화(Tokenization)
- 단어 임베딩(Word Embedding)
- 시퀀스 모델링

### 2. 문자 단위 RNN 분류기

**실습 파일 생성: `char_rnn_classification.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import string
import random

# 모든 문자와 언어 수
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 유니코드 문자열을 ASCII로 변환
def unicodeToAscii(s):
    return ''.join(
        c for c in s
        if c in all_letters
    )

# 문자를 인덱스로 변환
def letterToIndex(letter):
    return all_letters.find(letter)

# 문자를 텐서로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# 문자열을 텐서로 변환
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# RNN 모델 정의
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
      
        self.hidden_size = hidden_size
      
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
  
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
  
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# 샘플 데이터 생성
def create_sample_data():
    categories = ['English', 'Korean', 'Japanese']
  
    # 간단한 샘플 데이터
    samples = {
        'English': ['Smith', 'Johnson', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor'],
        'Korean': ['Kim', 'Lee', 'Park', 'Choi', 'Jung', 'Kang', 'Cho', 'Yoon'],
        'Japanese': ['Sato', 'Suzuki', 'Takahashi', 'Tanaka', 'Watanabe', 'Ito', 'Yamamoto', 'Nakamura']
    }
  
    return categories, samples

# 모델 생성
n_hidden = 128
categories, samples = create_sample_data()
n_categories = len(categories)

rnn = RNN(n_letters, n_hidden, n_categories)
print(f"RNN 모델 생성 완료! 카테고리 수: {n_categories}")
```

### 3. 텍스트 분류 모델 학습

**실습 파일 생성: `train_text_classifier.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from collections import Counter
import random
import time
import math

# 간단한 텍스트 분류기
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# 데이터 전처리
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 샘플 데이터 생성
def create_sample_text_data():
    # 간단한 감정 분석 데이터
    positive_texts = [
        "I love this movie it's amazing",
        "Great product highly recommend",
        "Excellent service very satisfied",
        "Perfect quality worth buying",
        "Wonderful experience will come back"
    ]
  
    negative_texts = [
        "Terrible movie waste of time",
        "Poor quality not recommended",
        "Bad service disappointed",
        "Awful product regret buying",
        "Horrible experience never again"
    ]
  
    # 레이블: 0=negative, 1=positive
    data = []
    for text in positive_texts:
        data.append((1, text))
    for text in negative_texts:
        data.append((0, text))
  
    return data

# 모델 학습 함수
def train_text_classifier():
    # 데이터 생성
    train_data = create_sample_text_data()
  
    # 어휘 구축
    vocab = Counter()
    for label, text in train_data:
        vocab.update(tokenizer(text))
  
    vocab_size = len(vocab)
  
    # 단어를 인덱스로 매핑
    word_to_idx = {word: i for i, word in enumerate(vocab)}
  
    # 모델 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(vocab_size, 64, 2).to(device)
  
    # 손실함수와 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
  
    # 학습 루프
    model.train()
    for epoch in range(10):
        total_loss = 0
        for label, text in train_data:
            # 텍스트를 인덱스로 변환
            text_indices = [word_to_idx.get(word, 0) for word in tokenizer(text)]
          
            text_tensor = torch.tensor(text_indices, dtype=torch.long).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)
            offsets = torch.tensor([0], dtype=torch.long).to(device)
          
            optimizer.zero_grad()
            output = model(text_tensor, offsets)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
          
            total_loss += loss.item()
      
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}')
  
    print("텍스트 분류기 학습 완료!")

# 학습 실행
train_text_classifier()
```

### 4. Transformer 기초

**실습 파일 생성: `transformer_basics.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
      
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
      
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
      
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
      
        self.register_buffer('pe', pe.unsqueeze(0))
      
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
      
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
      
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
      
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
      
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
      
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
      
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
      
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
      
        output = self.W_o(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
      
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
      
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
      
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
      
        return x

# 간단한 Transformer 모델
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
      
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
      
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
      
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
      
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
      
        x = self.ln(x)
        output = self.fc(x)
        return output

# 모델 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=100,
    dropout=0.1
).to(device)

print("Transformer 모델 생성 완료!")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
```

### 5. 간단한 언어 모델 학습

**실습 파일 생성: `language_model.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 간단한 언어 모델
class SimpleLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
      
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

# 샘플 텍스트 데이터
sample_text = """
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be that is the question.
All that glitters is not gold.
Where there is a will there is a way.
"""

# 텍스트 전처리
def preprocess_text(text):
    words = text.lower().split()
    # 특수 문자 제거
    words = [word.strip('.,!?";') for word in words]
    return words

# 어휘 구축
def build_vocab(words):
    vocab = list(set(words))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    return vocab, word_to_idx, idx_to_word

# 시퀀스 데이터 생성
def create_sequences(words, word_to_idx, seq_length):
    sequences = []
    for i in range(len(words) - seq_length):
        seq = [word_to_idx[word] for word in words[i:i+seq_length]]
        target = word_to_idx[words[i+seq_length]]
        sequences.append((seq, target))
    return sequences

# 언어 모델 학습
def train_language_model():
    # 데이터 전처리
    words = preprocess_text(sample_text)
    vocab, word_to_idx, idx_to_word = build_vocab(words)
  
    seq_length = 3
    sequences = create_sequences(words, word_to_idx, seq_length)
  
    # 모델 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLM(len(vocab), 64, 128).to(device)
  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
  
    # 학습 루프
    model.train()
    for epoch in range(100):
        total_loss = 0
        random.shuffle(sequences)
      
        for seq, target in sequences:
            seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            target_tensor = torch.tensor([target], dtype=torch.long).to(device)
          
            optimizer.zero_grad()
            output, _ = model(seq_tensor)
            loss = criterion(output[:, -1, :], target_tensor)
            loss.backward()
            optimizer.step()
          
            total_loss += loss.item()
      
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(sequences):.4f}')
  
    # 텍스트 생성 예제
    def generate_text(model, start_words, num_words):
        model.eval()
        words = start_words.split()
      
        with torch.no_grad():
            for _ in range(num_words):
                seq = [word_to_idx.get(word, 0) for word in words[-seq_length:]]
                seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
              
                output, _ = model(seq_tensor)
                predicted_idx = torch.argmax(output[:, -1, :], dim=-1).item()
                predicted_word = idx_to_word[predicted_idx]
              
                words.append(predicted_word)
      
        return ' '.join(words)
  
    # 텍스트 생성
    generated = generate_text(model, "the quick", 10)
    print(f"생성된 텍스트: {generated}")

# 학습 실행
train_language_model()
print("언어 모델 학습 완료!")
```

### 6. 실습 과제

1. **RNN 변형 실험**: GRU, LSTM 등 다른 순환 신경망으로 실험
2. **Transformer 실습**: 간단한 기계 번역 모델 구현
3. **언어 모델 개선**: 더 큰 데이터셋으로 언어 모델 학습

### 7. 추가 학습 자료

- [PyTorch 공식 튜토리얼](https://tutorials.pytorch.kr/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Papers with Code](https://paperswithcode.com/)

---

## 마무리

### 전체 프로그램 요약

1. **01차**: 딥러닝 서버 환경 설정 및 파이토치 기초
2. **02차**: CNN 및 컴퓨터 비전 모델 구현
3. **03차**: 생성 모델(DCGAN) 및 시계열 모델(LSTM)
4. **04차**: 자연어 처리 모델 및 Transformer 기초

### 다음 단계 학습 권장사항

1. **심화 프로젝트**: 배운 내용을 활용하여 개인 프로젝트 수행
2. **논문 리뷰**: 최신 딥러닝 논문 읽기 및 구현
3. **오픈소스 기여**: PyTorch 생태계 프로젝트 참여
4. **경진대회 참여**: Kaggle, DACON 등 데이터 과학 경진대회 참여

### 문의사항

- 강의자: 고원준 (1-2차), 안양준 (3-4차)
- 이메일: yangjunahn@sungshin.ac.kr
- 강의자료: [GitHub 저장소 URL]