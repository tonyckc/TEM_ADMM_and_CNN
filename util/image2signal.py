# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:26:38 2018

@author: ckc
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
curcwd=os.getcwd()
#target_path=curcwd+os.path.sep+"deblur_image"
#for image in os.listdir(target_path):
    #print("{}".format(image))
#im=Image.open(curcwd+os.path.sep+"deblur_image"+os.path.sep+"deblur_image_148.jpg")
#im=Image.open(curcwd+os.path.sep+"train_dataset"+os.path.sep+"real_dataset"+os.path.sep+'deblur_image_1700.jpg')
list_image = os.listdir('.//noise2raw_image')
for image_name in list_image:
    im = Image.open('.//noise2raw_image//' + image_name)
    arr_deblur=[]
    for i in range(30):
            for j in range(30):
                pixel = im.getpixel((j,i))
                #f.write('{}'.format(pixel))
                arr_deblur.append(pixel)
    arr_deblur_2 = np.zeros([30*30,1])

    for m in range(30*30):
            arr_deblur_2[m]=arr_deblur[m][0]*(256**0)+ arr_deblur[m][1]*(256**1)+ arr_deblur[m][2]*(256**2)
    plt.figure()
    plt.plot(np.linspace(1, 900, 899), arr_deblur_2[1:900], 'b-', label=image_name)
    plt.legend()
    plt.show()
'''
        arr_deblur_2[m]/=50
            #arr_deblur_2[m]=((arr_deblur_2[m] * 115000) + (2200*(16777216 - arr_deblur_2[m]))) / 16777216
        #arr_deblur_2=np.reshape(arr_deblur_2,(48*48,1))
    plt.figure()
    plt.plot(np.linspace(1,47,46),arr_deblur_2[1:47],'b-',label="image")
    plt.legend()
    plt.show()
    f.close()
'''