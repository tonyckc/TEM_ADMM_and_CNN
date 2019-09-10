import numpy as np
from PIL import Image

def get_next_batch(input_img,pointer, batch_size):
    img_batch = []
    imgs = input_img[pointer*batch_size:(pointer+1)*batch_size]
    for img in imgs:
        array = Image.open(img)
        array = np.array(array)
        array = array.astype('float32')/127.5 - 1
        img_batch.append(array)
    return img_batch