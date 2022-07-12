from datetime import datetime
start = datetime.now()
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision import models 
import torch
from scipy import ndimage

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

img = Image.open('test.jpeg')
trf = T.Compose([T.Resize(512), T.CenterCrop(508), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)

# Define the helper function
def decode_segmap(image, nc=21):
  label_colors = np.array([(0, 0, 0),
               (0, 0, 0), (0, 0, 0), (0,0, 0), (0, 0, 10), (0, 0, 0),
               (0, 0,0), (0,0,0), (0, 0, 0), (0, 0, 0), (0,0, 0),
               (0,0, 0), (0, 0, 0), (0, 0, 0), (0,0,0), (255, 255, 255),
               (0, 0, 0), (0,0, 0), (0, 0, 0), (0,0, 0), (0, 0,0)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb

out = fcn(inp)['out']
print (out.shape)

om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
rgb = decode_segmap(om) 
plt.imshow(rgb)
plt.show()

print('during running time: ', datetime.now() - start)