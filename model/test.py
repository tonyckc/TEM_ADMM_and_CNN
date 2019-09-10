# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:17:18 2019

@author: ckc
"""

import numpy as np
from PIL import Image

img1 = Image.open('0521MSE_10plainCNN2_25824_8.png')
img2 = np.array(img1).astype('float32')
img3 = np.clip(img2,0,255).astype(np.uint8)
img4 = Image.fromarray(img3)