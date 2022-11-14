import torch
from torchvision import models
from torchvision.models import ResNet50_Weights , resnet50



net = net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

print(net)
