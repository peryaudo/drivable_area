import torch
import torch.nn as nn

def get_model(train=True):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[-1] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))

    for param in model.classifier.parameters():
        param.requires_grad = train

    return model
