from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from model_repo import *
import _custom
import numpy as np
import os
import json
import time


train_loader, val_loader = _custom.dtcustom.custom_dtset()
test_data_iter = iter(val_loader)
test_image, test_label = test_data_iter.next()


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
imshow(torchvision.utils.make_grid(test_image))