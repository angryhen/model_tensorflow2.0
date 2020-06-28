import os
import cv2
import numpy as np
from skimage import io
from tqdm import tqdm

path1 = '/home/du/disk2/Desk/dataset/ibox/cls/dbox_c15/asm-asmnc-pz-yw-500ml/C-2020-05-11-09-1_asm-asmnc-pz-yw-500ml_000000.jpg'
path2 = '/home/du/disk2/Desk/dataset/ibox/cls/dbox_c15/hzy-hzy-pz-bxgw-500ml/2020-05-11-17-51-54_ty_ty-qbsds-pz-nmw-500ml_000000.jpg'
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
img1 = cv2.resize(img1, (224,224))
img2 = cv2.resize(img2, (224,224))
alpha = 1
lam = np.random.beta(alpha, alpha)
print(lam)
mixed_x = lam * img1 + (1 - lam) * img2
mixed_x = mixed_x.astype(np.uint8)
cv2.imshow('test', img1)
cv2.imshow('test1', img2)
cv2.imshow('test2', mixed_x)
cv2.waitKey(0)

