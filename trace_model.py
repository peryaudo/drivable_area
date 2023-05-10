import json
from matplotlib import pyplot as plt
import torch
from model import get_model
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
import coremltools as ct

class WrappedSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = get_model(train=False)
        checkpoint = torch.load("epoch_11.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def forward(self, x):
        return self.model(x)['out']

input_image = Image.open("/mnt/data/bdd100k/bdd100k/images/100k/val/bc7e7e2c-6388153c.jpg")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

traceable_model = WrappedSegmentationModel().eval()
trace = torch.jit.trace(traceable_model, input_batch)

scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
mlmodel = ct.convert(trace, inputs=[ct.ImageType(name="input", shape=input_batch.shape, scale=scale, bias=bias)])
mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps({"labels": ["direct", "alternative", "background"]})
mlmodel.save("segmentation.mlmodel")

# with torch.no_grad():
#     output = traceable_model(input_batch)[0]
# torch_predictions = output.argmax(0)
# plt.imshow(torch_predictions.numpy(), cmap='gray')
# plt.show()