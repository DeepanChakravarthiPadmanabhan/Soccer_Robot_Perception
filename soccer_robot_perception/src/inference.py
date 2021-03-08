import torch
import cv2
import numpy as np

from utils import plot_output

np.set_printoptions(suppress=True)

image_path = 
pathname = 

model = torch.load(pathname)

im = cv2.imread(image_path)
im = cv2.resize(im, (224, 224))
cv2.imshow('inp', im)
im = torch.Tensor(im).unsqueeze(0)
im = im.permute(0, 3, 1, 2)
print(im.shape)

out_det, out_seg = model(im)
plot_output(out_det, 'out_det', (56, 56))
plot_output(out_det, 'out_det', (224, 224))
plot_output(out_seg, 'out_seg', (56, 56))
plot_output(out_seg, 'out_seg', (224, 224))
cv2.destroyAllWindows()
