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
class ResNetBottom(torch.nn.Module):
    def __init__(self, original_model, real_name, num_feats):
        super(ResNetBottom, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        self.real_name = real_name
        self.num_feats = num_feats
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

res_model = torchvision.models.resnet50(pretrained=True)
res_conv2 = ResNetBottom(res_model, real_name="res52", num_feats=2048)
for param in res_conv2.parameters():
    param.requires_grad = False
res_conv2.eval()
model=res_conv2
x = torch.rand(1, 3, 224, 224)
torch_out = torch.onnx._export(model, x, "resnet50.onnx", export_params=True)

import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load("resnet50.onnx")
# convert model
model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"
print (model_simp)
onnx.save(model_simp,"resnet50-sim.onnx")