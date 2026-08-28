# 강의 계획 — CNN 모델 비교 및 DCGAN 실습

## 1. 오늘 수업 준비

### AlexNet, VGGNet-19, ResNet 비교

- 어제 해석한 VGGNet-19 코드 준비
- AlexNet 코드 준비
- ResNet 코드 준비
- `.ipynb` 파일에서 세 모델의 구조 비교
- AlexNet, VGGNet-19, ResNet의 차이 정리
- 교수에게 각 모델의 구조와 차이 직접 설명

### DCGAN 실습

- `img_alien_celeba.zip` 데이터 준비
- 로컬 컴퓨터에서 GPU 서버로 데이터 전송
- 서버에서 ZIP 파일 압축 해제
- PyTorch DCGAN Tutorial 웹페이지 준비
- Tutorial 코드를 따라 DCGAN 구현 및 실행

### Alien CelebA Dataset

데이터 다운로드:

https://www.kaggle.com/datasets/mrnobodyv/img-alien-celebazip?resource=download-directory

로컬 파일:

~~~text
/Users/yahn/Downloads/img_alien_celeba.zip
~~~

### PyTorch DCGAN Tutorial

https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html


## 2. AlexNet, VGGNet-19, ResNet 비교

### `.ipynb` 파일에서 모델 준비

~~~python
from torchvision import models

alexnet = models.alexnet()
vgg19 = models.vgg19()
resnet50 = models.resnet50()
~~~

AlexNet 구조:

~~~python
print(alexnet)
~~~

VGGNet-19 구조:

~~~python
print(vgg19)
~~~

ResNet 구조:

~~~python
print(resnet50)
~~~

### 각 모델에서 확인할 사항

- 전체 Layer 구성
- Convolution Layer 개수
- Kernel Size
- Stride
- Padding
- Channel 수의 변화
- Pooling Layer 위치
- Activation Function
- Fully Connected Layer 구조
- AlexNet과 VGGNet-19의 차이
- VGGNet-19와 ResNet의 차이
- ResNet의 Residual Block 및 Skip Connection


## 3. DCGAN 데이터 서버 전송

Mac 터미널에서 실행:

~~~bash
scp -i ~/.ssh/id_ed25519_team13_yangjun -P 22033 /Users/yahn/Downloads/img_alien_celeba.zip team13@210.125.91.90:/workspace/
~~~


## 4. GPU 서버 접속

~~~bash
ssh -i ~/.ssh/id_ed25519_team13_yangjun -p 22033 team13@210.125.91.90
~~~


## 5. 가상환경 활성화

~~~bash
cd /workspace
source .venv/bin/activate
~~~

Python 환경 확인:

~~~bash
which python
python --version
~~~


## 6. 데이터 확인 및 압축 해제

전송된 파일 확인:

~~~bash
ls -lh /workspace/img_alien_celeba.zip
~~~

압축 해제:

~~~bash
unzip img_alien_celeba.zip -d img_alien_celeba
~~~

압축 해제 결과 확인:

~~~bash
ls -al img_alien_celeba
~~~

하위 디렉터리 확인:

~~~bash
find img_alien_celeba -maxdepth 2 -type d
~~~


## 7. DCGAN Tutorial 실습

Tutorial:

https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html

### 코드 확인 순서

- Import
- Random Seed 설정
- Data Root 설정
- DataLoader
- 학습 이미지 확인
- Weight Initialization
- Generator 정의
- Discriminator 정의
- Loss Function
- Generator Optimizer
- Discriminator Optimizer
- Training Loop
- Loss 변화 확인
- 생성 이미지 확인
- 실제 이미지와 생성 이미지 비교


## 8. DCGAN 실행

Python 파일로 저장한 경우:

~~~bash
python py_dcgan.py
~~~


## 9. GPU 및 학습 상태 확인

GPU 확인:

~~~bash
nvidia-smi
~~~

GPU 지속 모니터링:

~~~bash
watch -n 1 nvidia-smi
~~~

실행 중인 학습 프로세스 확인:

~~~bash
ps aux | grep '[p]y_dcgan.py'
~~~


## 10. 학습 중단

터미널에서 직접 실행 중인 경우:

~~~text
Ctrl+C
~~~


## 11. nohup으로 학습 실행

~~~bash
nohup python py_dcgan.py > dcgan.log 2>&1 &
~~~

Process ID 확인:

~~~bash
echo $!
~~~

실시간 학습 로그 확인:

~~~bash
tail -f dcgan.log
~~~

로그 확인 종료:

~~~text
Ctrl+C
~~~


## 12. 학습 진행 상태 확인

프로세스 확인:

~~~bash
ps aux | grep 'py_dcgan.py'
~~~

GPU 확인:

~~~bash
nvidia-smi
~~~

최근 로그 확인:

~~~bash
tail -n 50 dcgan.log
~~~


## 13. 학습 종료 확인

~~~bash
ps aux | grep 'py_dcgan.py'
~~~

프로세스가 출력되지 않을 경우 마지막 로그 확인:

~~~bash
tail -n 50 dcgan.log
~~~