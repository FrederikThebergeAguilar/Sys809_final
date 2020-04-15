#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:03:11 2020

@author: fred
"""

from pathlib import Path
import cv2
import imutils
import numpy as np
import pickle
import random
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


dic = {      
           'Black':   0,
           'Cap':   1,
           'Damage':   2,
           'Good':   3,
           'Hair': 4
           }

def encode(label):
    return dic[label]

def decode(code):
    for pos in range(0 , 5):
        value = code[pos]
        if value == 1.0:
            return pos
        
def extract(image): 
    
    height, width = image.shape
    crop_img = image[470:height,0:width]
    crop_img2 = cv2.resize(crop_img, (400, 190))
    bitwise = cv2.bitwise_not(crop_img2) 
    equ = cv2.equalizeHist(bitwise)
    blur = cv2.medianBlur(equ, 5)
    
    kernel = np.ones((3,3),np.float32)/9
    sharpen = cv2.filter2D(blur,-1,kernel)
    
    img1 = cv2.Canny(sharpen,60,255)
  
    cnts = cv2.findContours(img1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    images = list()
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / (M["m00"]+1)))
        cY = int((M["m01"] / (M["m00"]+1)))
        shape = detect_shape(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        if(cY > 100 and shape != "unidentified"):
            #cv2.drawContours(crop_img, [c], -1, (0, 255, 0), 2)
            #cv2.putText(crop_img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(c)
            #start = x - 10 if x > 10 else 0
            #end = x + w + 10 if (x + w + 10) < 190 and start != 0 else x + w
            copy = crop_img.copy()
            roi = copy[0:1700, x*10: (x + w)*10]
            images.append(roi)
                
    return images

def detect_shape(contour):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) == 4:
        shape = "full item" 

    # otherwise, we assume the shape is a circle
    elif len(approx) == 2:
        shape = "partial item"

    # return the name of the shape
    return shape

def createDatabase():
    data_set = []
    data_dir = Path.cwd() / "images" 
    for first_level in data_dir.glob('*'):
        if first_level.is_dir():
            label = ((str(first_level).split('/'))[-1])
            print(label)
            pos = 0
            for imagePath in first_level.glob('*'):
                label = ((str(first_level).split('/'))[-1])
                img = cv2.imread(str(imagePath))
                images = extract(img)
                
                for image in images:
                    #save image pickle or file ?
                    name = "crop_"+label
                    image_path = Path.cwd() / name
                    name = str(pos) +"_"+name+".jpg"
                    saveImages(image,image_path,name)
                    pos = pos + 1
def augmente_data():
    data_dir = Path.cwd() / "extracted_images"
    index = 0
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,zca_whitening=True)
    for first_level in data_dir.glob('*'):
        if first_level.is_dir():
            
            label = ((str(first_level).split('_'))[-1])
            print(label)
            for imagePath in first_level.glob('*'):
                img = cv2.imread(str(imagePath))
                data = img_to_array(img)
                samples = np.expand_dims(data, 0)
                it = datagen.flow(samples, batch_size=1)
                # configure batch size and retrieve one batch of images
                dir_name = "aug_"+label
                target_dir = Path.cwd() / "augmented_images" / dir_name
                for i in range(0, 9):
                        name = "aug"+str(index)+".png" 
                        batch = it.next()
                        image = batch[0].astype('uint8')
                        saveImages(image,target_dir,name)
                        index = index + 1 
                

    return 0
               
def readFiles():
    data_set = []
    data_dir = Path.cwd() / "augmented_images" 
    for first_level in data_dir.glob('*'):
        if first_level.is_dir():
            label = ((str(first_level).split('_'))[-1])
            print(label)
            for imagePath in first_level.glob('*'):
                img = cv2.imread(str(imagePath))
                resize_image = cv2.resize(img, (25, 95))
                data_set.append([resize_image, label])

    features = []
    labels = []
    random.shuffle(data_set)
    for feature, label in data_set:
        features.append(feature)
        labels.append(encode(label))

    pickle_out = open("features.pickle", "wb")
    pickle.dump(features, pickle_out)
    pickle_out.close()

    pickle_out = open("labels.pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()
    return np.array(features),labels

def saveImages(image,outputPath,name):
    try: 
        filename = outputPath / name
        cv2.imwrite(str(filename), image)
        print(str(filename))
    except:        
        print(name+" fail")
    