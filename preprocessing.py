import cv2 as cv
import os
import random as r 
from PIL import Image
from numpy import asarray

path = r'INSERT_PATH'

#resizing images with bicubic interpolation
for file in os.listdir(path):
        image = Image.open(path + file)
        width = 512
        height = 512
        img = image.resize((height, width), Image.BICUBIC)
        img.save(path + file)

#apply Laplace Operator  
for count, filename in enumerate(os.listdir(path)):
    image = cv.imread(path + '/' + filename + '.png')
    LP = cv.Laplacian(image,cv.CV_64F)
    cv.imwrite(path + '/' + filename + '.png', LP)

#affine transformations    
for x, filename in enumerate(os.listdir('INSERT PATH')):
    image = Image.open('INSERT_PATH' + filename)
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
