# -*- coding: utf-8 -*-

"""
Created on Mon Jul 16 00:56:12 2018

@author: kecheng chen
"""

import datetime
import numpy as np
import sys
sys.path.insert(0, './util')
import tensorflow as tf
import os
from PIL import Image
import scipy.misc as misc
import random
import create_save_folder
import get_img
import get_next_batch
from argparse import ArgumentParser
from .model import optimize

training_id = 'ADMM0910'
MODEL_SAVE_PATH = './logs_{}'.format(training_id)
TENSORBOARD_SAVE_PATH = './tensorboard_{}'.format(training_id)
DENOISING_IMG_PATH = './denoising_img_{}'.format(training_id)
DATASET_PATH_INPUT = './src/noise_image'
DATASET_PATH_REAL = './src/noise2raw_image'
DATASET_PATH_TEST = './src/test'
LEARNING_RATE_BASE = 1e-4
LEARNING_RATE_DECAY = 0.95
EPOCHS = 5
LOAD_MODEL = False
TRAIN_MODEL = True
BATCH_SIZE = 1
IMG_SIZE = 30
DEVICE = '/gpu:0'


def main():
    create_save_folder(MODEL_SAVE_PATH)
    create_save_folder(TENSORBOARD_SAVE_PATH)
    create_save_folder(DENOISING_IMG_PATH)

    input_data = get_img(DATASET_PATH_INPUT)
    real_data = get_img(DATASET_PATH_REAL)
    test_data = get_img(DATASET_PATH_TEST)

    optimize.train(input_data, real_data, test_data, training_id, MODEL_SAVE_PATH, TENSORBOARD_SAVE_PATH,
          DENOISING_IMG_PATH, LEARNING_RATE_BASE, LEARNING_RATE_DECAY, EPOCHS, LOAD_MODEL=False, BATCH_SIZE=BATCH_SIZE,
          IMG_SIZE=IMG_SIZE, DEVICE=DEVICE)

if __name__ == '__main__':
     main()







