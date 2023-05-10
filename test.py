from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from dataset import DrivableAreaDataset, get_transforms
from model import get_model
from torch.utils.data import DataLoader

model = get_model(train=False)
model.eval()
checkpoint = torch.load("epoch_11.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cuda')

dataset = DrivableAreaDataset(root='/mnt/data/bdd100k/bdd100k', phase='val', transforms=get_transforms())

dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=8)

if True:
    images, targets = next(iter(dataloader))
    images = images.to('cuda')
    targets = targets.to('cuda')
    outputs = model(images)['out']
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu()
    images = images.cpu()
    targets = targets.cpu()
    for image, pred, target in zip(images, preds, targets):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
        for t, m, s in zip(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        ax[0].imshow(image.numpy().transpose(1, 2, 0))
        ax[1].imshow(pred.numpy(), cmap='gray')
        ax[2].imshow(target.numpy(), cmap='gray')
        plt.show()
    exit(0)

acc = 0

with torch.no_grad():
    for images, targets in tqdm(dataloader):
        images = images.to('cuda')
        targets = targets.to('cuda')
        outputs = model(images)['out']

        _, preds = torch.max(outputs, 1)
        correct_pixels = (preds == targets).sum().item()
        total_pixels = images.size(0) * images.size(2) * images.size(3)
        batch_acc = correct_pixels / total_pixels
        acc += batch_acc

acc /= len(dataloader)

print("Test Acc: %.4f" % (acc))