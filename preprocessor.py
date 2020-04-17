#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:00:36 2020

@author: rahulkumar
"""

import cv2 as cv
import os
import random as r 
from PIL import Image

path = r'INSERT PATH'
     
for count, filename in enumerate(os.listdir(path)): 
    dst = str(r.randint(1,100000) * r.randint(1,100000) * r.randint(1,100000)) + ".png"
    src = path + '/' + filename 
    dst = path + '/' + dst 
          
        # rename() function will 
        # rename all the files 
    os.rename(src, dst)

  
for count, filename in enumerate(os.listdir(path)):
    image = cv.imread(path + '/' + filename + '.png')
    LP = cv.Laplacian(image,cv.CV_64F)
    cv.imwrite(path + '/' + filename + '.png', LP)
    
for x, filename in enumerate(os.listdir('INSERT PATH')):
    image = Image.open('INSERT PATH' + filename)
    r0 = image.rotate(0)
    r5 = image.rotate(5)
    r10 = image.rotate(10)
    r15 = image.rotate(15)
    r20 = image.rotate(20)
    r25 = image.rotate(25)
    
    r0.save('INSERT PATH' + str(r.randint(1,100000) * r.randint(1,100000) * r.randint(1,100000)) + '.png' , 'PNG')
    r5.save('INSERT PATH' + str(r.randint(1,100000) * r.randint(1,100000) * r.randint(1,100000)) + '.png' , 'PNG')
    r10.save('INSERT PATH' + str(r.randint(1,100000) * r.randint(1,100000) * r.randint(1,100000)) + '.png' , 'PNG')
    r15.save('INSERT PATH' + str(r.randint(1,100000) * r.randint(1,100000) * r.randint(1,100000)) + '.png' , 'PNG')
    r20.save('INSERT PATH' + str(r.randint(1,100000) * r.randint(1,100000) * r.randint(1,100000)) + '.png' , 'PNG')
    r25.save('INSERT PATH' + str(r.randint(1,100000) * r.randint(1,100000) * r.randint(1,100000)) + '.png' , 'PNG')

    