import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR
MODEL_PATH = BASE_DIR / "models" / "alexnet_cifar10.pth"

NUM_IMAGES = 16
NUM_CLASSES = 10
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
print(f"Model path: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = AlexNet(num_classes=checkpoint.get("num_classes", NUM_CLASSES))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

if "train_accuracy" in checkpoint:
    print(f"Checkpoint train accuracy: {checkpoint['train_accuracy']:.2f}%")
if "test_accuracy" in checkpoint:
    print(f"Checkpoint test accuracy: {checkpoint['test_accuracy']:.2f}%")

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

display_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=None,
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=test_transform,
)

indices = torch.randperm(len(test_dataset))[:NUM_IMAGES].tolist()

images = []
labels = []

for index in indices:
    image, label = test_dataset[index]
    images.append(image)
    labels.append(label)

images = torch.stack(images).to(device)
labels = torch.tensor(labels, device=device)

with torch.no_grad():
    logits = model(images)
    probabilities = F.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(1)

rows = 4
cols = 4
figure, axes = plt.subplots(rows, cols, figsize=(12, 12))

for i, axis in enumerate(axes.flatten()):
    raw_image, true_label = display_dataset[indices[i]]
    predicted_label = predictions[i].item()
    confidence = confidences[i].item() * 100.0
    is_correct = predicted_label == true_label

    axis.imshow(raw_image)
    axis.axis("off")
    axis.set_title(
        f"Pred: {CIFAR10_CLASSES[predicted_label]} ({confidence:.1f}%)\n"
        f"True: {CIFAR10_CLASSES[true_label]}",
        color="green" if is_correct else "red",
        fontsize=9,
    )

figure.suptitle("CIFAR-10 Predictions", fontsize=14)
figure.tight_layout()
plt.show()
