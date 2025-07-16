# Summer Class - Deep Learning Tutorial

## Overview

This repository contains materials and tutorials for the Summer Class Deep Learning program at Sungshin Women's University. The program focuses on practical deep learning implementation using PyTorch and GPU servers.

## Program Details

- **Duration**: 4 days (8 hours total)
- **Schedule**: August 18-22, 2025
- **Target Audience**: Students new to deep learning servers
- **Platform**: Windows/Mac users

## Course Structure

| Session | Date | Time | Instructor | Topic |
|---------|------|------|------------|-------|
| 01 | Aug 18 (Mon) | 10:00-12:00 | Ko Wonjun | Deep Learning Server Usage & PyTorch Neural Network Basics |
| 02 | Aug 19 (Tue) | 10:00-12:00 | Ko Wonjun | CNN Models & Object Detection Fundamentals |
| 03 | Aug 21 (Thu) | 10:00-12:00 | Ahn Yangjun | DCGAN & Time Series Models |
| 04 | Aug 22 (Fri) | 10:00-12:00 | Ahn Yangjun | NLP Models & Examples |

## Server Information

- **IP Address**: 210.125.91.90
- **Port**: 22
- **Username**: yangjunahn
- **Password**: 0000
- **Available GPUs**: 0-7 (8 total)

## Prerequisites

### Required Software

#### Windows Users
- PuTTY or Windows Terminal
- Visual Studio Code
- Git (optional)

#### Mac Users
- Terminal (pre-installed)
- Visual Studio Code
- Git (optional)

### VS Code Extensions
- Remote - SSH
- Python
- Jupyter

## Getting Started

### 1. Server Connection

#### Using SSH Client

**Windows (PuTTY)**
1. Launch PuTTY
2. Host Name: `210.125.91.90`
3. Port: `22`
4. Connection type: SSH
5. Click Open
6. Username: `yangjunahn`
7. Password: `0000`

**Mac/Linux**
```bash
ssh yangjunahn@210.125.91.90
```

#### Using Visual Studio Code
1. Open VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
3. Select "Remote-SSH: Connect to Host"
4. Enter `yangjunahn@210.125.91.90`
5. Enter password: `0000`

### 2. Environment Setup

```bash
# Create working directory
mkdir -p ~/pytorch_tutorial
cd ~/pytorch_tutorial

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. GPU Usage

#### Check GPU Status
```bash
nvidia-smi
```

#### GPU Assignment
Students are assigned GPUs as follows:
- Students 1-2: GPU 0
- Students 3-4: GPU 1
- Students 5-6: GPU 2
- Students 7-8: GPU 3
- Students 9-10: GPU 4
- Students 11-12: GPU 5
- Students 13-14: GPU 6
- Students 15-16: GPU 7

#### Specify GPU in Code
```python
import torch
import os

# Specify GPU (e.g., GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## Course Materials

### Session 1: PyTorch Basics
- Tensor operations
- Neural network fundamentals
- FashionMNIST dataset handling

### Session 2: CNN & Object Detection
- Convolutional Neural Networks
- Image classification
- Object detection basics

### Session 3: Advanced Models
- DCGAN implementation
- Time series models
- Generative models

### Session 4: Natural Language Processing
- Text processing
- NLP model architectures
- Practical applications

## Repository Structure

```
summer_class/
├── tutorials/
│   ├── session1/
│   ├── session2/
│   ├── session3/
│   └── session4/
├── examples/
├── datasets/
└── README.md
```

## Contributing

This repository is for educational purposes. Students are encouraged to:
- Follow along with tutorials
- Experiment with provided code
- Ask questions and seek help when needed

## Contact

For questions about the course materials or technical issues:
- **Instructor**: Ko Wonjun, Ahn Yangjun
- **Course**: AI Convergence Department, Sungshin Women's University

## License

This educational material is provided for the Summer Class program. Please respect the academic integrity of the course.