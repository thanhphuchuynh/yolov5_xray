import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os

from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2
dicom_image = '/Final/Dicom/vinbigdata/7c6f191b5d28bc1992e491d906f0d1a5.dicom'
dicom = pydicom.read_file(dicom_image)
print(dicom)


data_voi = apply_voi_lut(dicom.pixel_array, dicom)

data = dicom.pixel_array
data_voi = data_voi - np.min(data_voi)
data_voi = data_voi / np.max(data_voi)
data_voi = (data_voi * 255).astype(np.uint8)

print(data_voi)

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im


plt.figure(figsize = (12,12))
plt.imshow(data_voi, 'gray')
# plt.show()
filename = 'savedImage.jpg'
# plt.figure(figsize = (12,12))
# plt.imshow(resize(data_voi,512), 'gray')
# plt.show()
plt.savefig('foo.png')
cv2.imwrite(filename, data_voi)