# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:21:15 2018

@author: ckc
"""
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio#io相关模块，进行操作。
import random
# crate floder
curcwd = os.getcwd()
if os.path.exists('.//noise_image'):
    print('noise_image floder exists')
else:
    os.makedirs('.//noise_image')
if os.path.exists('.//noise2raw_image'):
    print('noise2raw_image floder exists')
else:
    os.makedirs('.//noise2raw_image')

#actual_detection_signal

mat_theory = 'clean_signal.mat'
mat_noise = 'noise_signal.mat'
#mat_theory='matlab.mat'

data_theory = sio.loadmat(mat_theory)
data_noise = sio.loadmat(mat_noise)
# load_matrix=data_theory['xxx']

load_matrix_theory = data_theory['clean_sig']
load_matrix_noise = data_noise['noise_sig']

# use random generate rand array
rand_arr = random.sample(range(1, 100000), 99999)

num = 1
for rand_num in rand_arr:
    # load rand data and shape trans
    signal_theory = load_matrix_theory[rand_num]
    signal_noise = load_matrix_noise[rand_num]

    signal_theory = np.reshape(signal_theory, (900, 1))
    signal_noise = np.reshape(signal_noise, (900, 1))

    signal_th_trans = np.reshape(signal_theory, (30, 30))
    signal_no_trans = np.reshape(signal_noise, (30, 30))
    # create image
    img_theory = Image.new('RGB', (30, 30))
    img_noise = Image.new('RGB', (30, 30))

    name1 = '{}.png'.format(num)
    name2 = '{}noise.png'.format(num)
    path1 = curcwd+os.path.sep+'noise2raw_image'+os.path.sep+name1
    path2 = curcwd+os.path.sep+'noise_image'+os.path.sep+name2
    # put pixel
    for i in range(30):
        for j in range(30):
            img_theory.putpixel((j, i), int(signal_th_trans[i][j]))
            img_noise.putpixel((j, i), int(signal_no_trans[i][j]))
    # save image
    img_theory.save(path1)
    img_noise.save(path2)
    print('The {}th pair images are saved!\n '.format(num))
    num += 1