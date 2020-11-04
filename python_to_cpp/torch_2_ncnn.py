import torch
import torchvision
import re
import torch
import os
import random
import torch
import torchvision
import torch.onnx

# https://github.com/daquexian/onnx-simplifier

model = torchvision.models.resnet50(pretrained=True)
model.eval()
# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 224, 224)
# Export the model
torch_out = torch.onnx._export(model, x, "resnet50.onnx", export_params=True)