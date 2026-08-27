# 1. 실험 코드의 수정

## 1.1. 문제점 목록

### 1. 가중치 초기화(Weight Initialization) 표준편차 조정

AlexNet 원본의 $\mathcal{N}(0, 0.01^2)$ 초기화는 ImageNet(입력 차원 $224 \times 224$) 기준 설정입니다. CIFAR-10용 축소 구조에서는 초기 레이어 및 선형 레이어의 출력 분산이 급격히 소실되어 역전파 시 그래디언트 소실이 발생합니다.

* `nn.init.normal_`의 표준편차를 0.01 대신 **0.05 ~ 0.1** 수준으로 상향하거나, AlexNet 당시 도입된 균등 분포/분산 스케일링 방식을 적용합니다.
* 편향(bias)을 1로 초기화하는 설정(ReLU 활성화 유도 목적)이 현재 얕은 CIFAR 특징 맵에서는 특정 뉴런의 출력을 포화시킬 수 있으므로 **모든 bias를 0으로 초기화**하여 안정성을 확보합니다.

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

```

AlexNet 원본 논문(ImageNet Classification with Deep Convolutional Neural Networks, Krizhevsky et al., 2012) Section 5. Details of learning에는 가중치 초기화에 대해 다음과 같이 명시되어 있습니다:

"We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01. We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs. We initialized the neuron biases in the remaining layers with the constant 0."  

즉, 분산 스케일링(Glorot/Xavier 또는 He 초기화) 및 균등 분포 방식은 AlexNet 원본 논문에서 사용된 방식이 아니며, 당시(2010년 Xavier Glorot의 연구 등) 제안되어 널리 쓰이던 방식으로 구현한 상태입니다. 따라서 논문 원본의 수식인 평균 0, 표준편차 0.01의 가우시안 정규분포(및 특정 레이어 편향 1)로 설정해야 합니다.

---

### 2. Local Response Normalization (LRN) 적용

AlexNet 논문(Krizhevsky et al., 2012)에서 Batch Normalization 등장 이전에 측면 억제(Lateral Inhibition)를 구현하기 위해 제안한 핵심 정규화 기법입니다.

* `Conv2d` $\rightarrow$ `ReLU` 직후, `MaxPool2d` 이전에 `nn.LocalResponseNorm`을 배치합니다.

```python
# features 내부 1, 2번째 Conv 블록에 적용 예시
nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
nn.ReLU(inplace=True),
nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
nn.MaxPool2d(kernel_size=2, stride=2),

nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
nn.ReLU(inplace=True),
nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
nn.MaxPool2d(kernel_size=2, stride=2),

```

#### LRN의 물리적/생물학적 메커니즘: 측면 억제 (Lateral Inhibition)

LRN(Local Response Normalization)의 생물학적 기원은 신경생리학의 측면 억제(Lateral Inhibition)입니다.

실제 망막 신경절 세포나 대뇌 시각 피질(V1)의 뉴런은 강한 시각 자극을 받아 흥분할 때, 인접한 주변 뉴런들로 억제성 신호를 전달해 이웃 뉴런의 반응을 감쇄시킵니다.

$$b_{x, y}^{i} = \frac{a_{x, y}^{i}}{\left( k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x, y}^{j})^2 \right)^\beta}$$

* **$a_{x, y}^{i}$**: 공간 좌표 $(x, y)$, 커널 $i$에서 $\text{ReLU}$를 통과한 활성화 값
* **분모의 합산 항**: 동일 좌표 $(x, y)$에서 인접한 $n$개 채널들의 에너지 합
* **물리적 효과**: 특정 채널 $i$의 신호가 독보적으로 클 경우, 인접 채널들의 분모를 증가시켜 상대적으로 약한 신호를 억제하고 가장 두드러진 피처 신호의 국소적 대비(Contrast)를 증폭합니다.

---

#### 하나 더 알기. 마흐 밴드(Mach Band) 착시

마흐 밴드 현상은 인간 시각계의 측면 억제 메커니즘에 의해 발생하는 대표적인 착시 현상입니다.

* **마흐 밴드 착시의 원리**: 밝기가 계단식 또는 점진적으로 변하는 경계면에서, 망막의 측면 억제로 인해 밝은 영역의 경계선 바로 안쪽은 실제보다 더 밝게 보이고, 어두운 영역의 경계선 바로 바깥쪽은 실제보다 더 어둡게 인식됩니다(오버슈트/언더슈트 현상).
* **LRN과의 대응 관계**:
* 마흐 밴드는 **공간 영역(Spatial Domain)** 상의 인접 수용야(Receptive Field) 간 측면 억제로 인해 경계면(Edge) 대비가 극대화되는 현상입니다.
* LRN은 이를 채널/특징 영역(Channel/Feature Domain)으로 확장하여, 동일한 공간 위치에서 서로 다른 특징(예: 서로 다른 방향의 에지 커널 등) 간에 경쟁(Competition)을 유도하고 가장 지배적인 특징 반응을 선명하게 부각시킵니다.

---

### 3. 완전연결계층(FC Layer) 차원 최적화

AlexNet의 FC 층 뉴런 수(4096개)는 CIFAR-10 데이터셋 크기(50,000장) 대비 파라미터가 과도하여 최적화 초기 단계에서 학습 정체를 유발합니다. 2012년 당시 작은 데이터셋에 적용하던 서브 네트워크 구조로 축소합니다.

* FC 층의 뉴런 수를 **4096 $\rightarrow$ 1024 또는 512** 수준으로 축소하여 그래디언트 흐름을 개선합니다.

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256 * 4 * 4, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, num_classes)
)

```

---

### 4. 에포크(Epochs) 및 학습률 스케줄러(Scheduler) 조정

현재 설정된 `EPOCHS = 5`는 SGD 옵티마이저로 수렴하기에 절대적으로 부족한 반복 횟수입니다. 또한 `StepLR(step_size=3, gamma=0.1)`은 3에포크 만에 학습률을 0.001로 급감시켜 모델이 학습을 멈추게 만듭니다.

* 학습 에포크를 **90 ~ 100 Epoch** 이상으로 확장합니다.
* 학습률 스케줄러의 감쇠 주기를 전체 에포크 길이에 맞게 분배합니다.

```python
EPOCHS = 90
LEARNING_RATE = 0.01

# 30 에포크마다 0.1배 감쇠
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)

```

---

### 5. 고전적 데이터 증강(Data Augmentation) 강화

AlexNet 원본 논문에서 사용된 표준 증강 방식(Random Crop & Horizontal Flip)을 CIFAR-10 입력에 적용합니다.

* `transforms.Resize((32, 32))` 대신 패딩 후 무작위 자르기를 적용하여 위치 불변성을 확보합니다.

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

```

## 1.2. 수정 방안

### 1.2.1. AlexNet의 구조를 따라가기: 가중치 초기화 방식과 LRN 적용

우선 가중치 초기화 방법을 AlexNet의 제안과 다르게 현재의 작은 CIFAR10 데이터에 맞춰 변경해 봅니다. Bias를 1이 아닌 0으로 수정했어요. 그리고 기존 코드에 빠져 있던 LRN을 추가합니다. 

py_alexnet_v2.py

### 1.2.2. 추가 수정: FC Layer 차원, 학습 에폭의 수정, 데이터 증강

그리고 FC Layer 차원 수정, 학습 에폭을 늘리고 학습률 감쇠를 적절히 변경한 뒤, 데이터 증강도 적용합니다. 

py_alexnet_v3.py

### 1.2.3. 추가 수정: 

py_alexnet_v4.py

원인 분석: 합성곱 계층에서의 분산 폭발 (Variance Explosion)

가중치 표준편차를 `0.05`로 일괄 상향 적용할 경우 선형 계층(Linear)은 안정되지만, 5개의 합성곱 계층(Conv2d)을 통과하면서 채널 수 누적에 따라 출력 분산이 지수적으로 폭발합니다.

$$\text{Var}(\text{Conv output}) \propto c_{\text{in}} \times k^2 \times \sigma^2$$

* 첫 번째 Conv ($c_{\text{in}}=3, k=5$): $3 \times 25 \times 0.05^2 = 0.1875$
* 두 번째 Conv ($c_{\text{in}}=96, k=5$): $96 \times 25 \times 0.05^2 = 6.0$ (분산 급증)
* 세 번째 Conv ($c_{\text{in}}=256, k=3$): $256 \times 9 \times 0.05^2 = 5.76$

이로 인해 LRN을 거치더라도 심층 Conv 레이어의 출력이 포화 및 발산하여, `lr=0.01` 환경에서 첫 에폭부터 그래디언트가 발산(Exploding Gradient)하거나 로짓이 무너져 예측 확률이 $0.10$에 고정됩니다.

---

### 수정 대상 라인 및 변경 방법

#### 1. Conv와 Linear 계층의 초기화 표준편차 분리 (Line 108, Line 114)



합성곱 계층의 팬인(Fan-in) 규모를 고려하여 Conv2d는 `0.01`, Linear 계층은 `0.05`로 각각 초기화합니다.

* **Line 108**: `nn.init.normal_(layer.weight, mean=0.0, std=0.01)`로 수정


* **Line 114**: `nn.init.normal_(layer.weight, mean=0.0, std=0.05)` 유지



```python
    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.05)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

```

#### 2. Dropout 비율 완화 (Line 92, Line 96)



신호 전파 초기에 정보 유실을 줄이기 위해 드롭아웃 확률을 `0.5`에서 `0.2`로 완화합니다.

```python
# Line 91 ~ 101 수정
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, num_classes)
        )

```

#### 3. Learning Rate 상향 (Line 29)

초기 수렴 가속 및 안장점 탈출을 위해 학습률을 `0.01`에서 `0.05` 또는 `0.1`로 상향합니다.

```python
LEARNING_RATE = 0.05

```

# 2. Hyperparameter Optimization

---

## 1. 하이퍼파라미터의 중요성

딥러닝 모델의 파라미터는 역전파(Backpropagation)를 통해 데이터로부터 자동으로 학습되는 내부 가중치(Weights & Biases)와 학습 과정 자체를 제어하는 하이퍼파라미터(Hyperparameters)로 구분됩니다.

* **손실 함수 평면(Loss Surface)의 지형 변화**: 가중치 초기화 편차($\sigma$), 정규화(LRN, Dropout) 계수 등은 최적화 알고리즘이 탐색하는 손실 평면의 곡률과 기울기를 근본적으로 결정합니다.
* **수렴성과 안정성 결정**: 학습률(Learning Rate), 모멘텀(Momentum), 배치 크기(Batch Size)의 상호작용에 따라 모델이 전역/국소 최적점(Global/Local Minima)에 도달하거나, 그래디언트 소실/발산으로 인해 학습이 완전히 붕괴될 수 있습니다.
* **일반화 성능(Generalization Capability)**: 가중치 감쇠(Weight Decay) 및 드롭아웃 비율(Dropout Rate)은 과적합(Overfitting)을 억제하고 검증 데이터에 대한 최종 예측 정확도를 좌우합니다.

---

## 2. Optuna 개요 및 핵심 논문

**Optuna**는 하이퍼파라미터 탐색 공간을 효율적으로 자동화하기 위해 설계된 차세대 베이지안 최적화(Bayesian Optimization) 프레임워크입니다.

* **논문 링크**: [Optuna: A Next-generation Hyperparameter Optimization Framework (KDD 2019)](https://arxiv.org/abs/1907.10902)

### 핵심 설계 특징

1. **Define-by-Run API**: 정적 선언 방식 대신 Python의 일반 제어문(if, for) 내에서 동적으로 탐색 공간을 정의합니다.
2. **효율적인 샘플링(Sampling)**: Tree-structured Parzen Estimator (TPE) 알고리즘을 기반으로 과거 평가 결과를 반영하여 유망한 파라미터 영역을 집중 탐색합니다.
3. **효과적인 가지치기(Pruning)**: 학습 곡선의 초기 성능이 낮을 경우 에포크 도중 시도(Trial)를 조기 종료(Early Stopping)하여 연산 자원을 절약합니다.

---

## 3. Optuna 구현 구조 및 워크플로우

```
                ┌────────────────────────────────┐
                │          optuna.Study          │
                │   (전체 최적화 프로세스 관리)    │
                └───────────────┬────────────────┘
                                │ creates
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                      optuna.Trial                         │
  │                                                           │
  │  1. suggest_float / suggest_int / suggest_categorical     │
  │     (하이퍼파라미터 샘플링)                               │
  │                                                           │
  │  2. Model Training & Validation                           │
  │                                                           │
  │  3. trial.report() -> trial.should_prune()                │
  │     (중간 검증값 보고 및 조기 종료 판단)                  │
  │                                                           │
  │  4. return validation_metric                              │
  └───────────────────────────────────────────────────────────┘

```

---

## 4. PyTorch 기반 실습 코드

```python
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_model(trial):
    """하이퍼파라미터 샘플링을 기반으로 네트워크 생성"""
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    fc_units = trial.suggest_categorical("fc_units", [512, 1024])

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(64 * 8 * 8, fc_units),
        nn.ReLU(),
        nn.Linear(fc_units, 10),
    )
    return model


def objective(trial):
    """Optuna가 반복 실행할 목적 함수"""
    # 1. 하이퍼파라미터 샘플링
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    # 2. 데이터셋 설정
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    full_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    full_test = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # 빠른 탐색을 위해 서브셋 사용 (실제 실험 시 전체 데이터 활용)
    train_subset = Subset(full_train, range(5000))
    val_subset = Subset(full_test, range(1000))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = build_model(trial).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )

    # 3. 모델 학습 및 검증
    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 검증 정확도 측정
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = correct / total

        # 4. Pruning(가지치기)을 위한 중간 결과 보고
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    # 최대화(maximize) 목표 설정 및 TPE 샘플러, MedianPruner 구성
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    study.optimize(objective, n_trials=30, timeout=1200)

    print(f"Best Trial Accuracy: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

```