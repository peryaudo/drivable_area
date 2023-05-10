import os
import pdb
from typing import Any
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import DrivableAreaDataset, get_transforms
from model import get_model

dataset = DrivableAreaDataset(root='/mnt/data/bdd100k/bdd100k', phase='train', transforms=get_transforms())

dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=8)

model = get_model(train=True)
checkpoint = torch.load("epoch_6_2051.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')

num_epochs = 25
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5, num_epochs):
    print('Epoch %d/%d' % (epoch, num_epochs - 1))

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    tqdm_dataloader = tqdm(dataloader)
    for i, (images, targets) in enumerate(tqdm_dataloader):
        images = images.to('cuda')
        targets = targets.to('cuda')

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(images)['out']
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_pixels = (preds == targets).sum().item()
            total_pixels = images.size(0) * images.size(2) * images.size(3)
            batch_acc = correct_pixels / total_pixels
            epoch_acc += batch_acc
            tqdm_dataloader.set_postfix_str('batch_acc: %.4f loss: %.4f' % (batch_acc, loss.item()))
        
    epoch_loss /= len(dataloader)
    epoch_acc /= len(dataloader)
    print("Loss: %.4f, Acc: %.4f" % (epoch_loss, epoch_acc))

    checkpoint_path = f"epoch_{epoch + 1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print('Saved checkpoint to %s' % checkpoint_path)

# print(len(model.parameters()))

# print (imgs.shape)

# imgs, targets = next(iter(dataloader))
# imgs = imgs.to('cuda')
# with torch.no_grad():
#     output = model(imgs)['out'][0]
# output_predictions = output.argmax(0)
# output_predictions = output_predictions.to('cpu').numpy()
# plt.imshow(output_predictions, cmap='gray')
# plt.show()
# pdb.set_trace()

# print(imgs[0].shape)
# print(targets[0].shape)

# plt.imshow(imgs[0].permute(1, 2, 0))
# plt.imshow(targets[0] == 2, cmap='gray')
# plt.show()

# t = targets[0].detach().numpy()
# Print all unique values in an numpy array t
# print(np.unique(t))