import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

from model import get_model

reader = imageio.get_reader('IMG_1813.MOV')

# model_input_shape = transforms.ToTensor()(Image.open("/mnt/data/bdd100k/bdd100k/images/100k/val/bc7e7e2c-6388153c.jpg")).shape
# print(model_input_shape)

model = get_model(train=False).to("cuda")
checkpoint = torch.load("epoch_22.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(720, 1280)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

with imageio.get_writer('output_video.mp4', fps=30) as writer:
    for i, frame in enumerate(tqdm(reader, total=19812)):
        # print(f"processing frame {i}...")
        frame_image = Image.fromarray(frame)
        input_tensor = preprocess(frame).to("cuda")
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch)['out'][0]
        preds = output.argmax(0)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([2, 1, 0])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        r = Image.fromarray(
            preds.byte().cpu().numpy()
        ).resize(frame_image.size)
        r.putpalette(colors)
        r = r.convert("RGBA")
        r.putalpha(128)
        alpha_image = frame_image.copy()
        alpha_image.putalpha(255)
        result_image = Image.alpha_composite(alpha_image, r)
        # plt.figure(dpi=240)
        # plt.imshow(result_image)
        # plt.show()
        writer.append_data(np.array(result_image))