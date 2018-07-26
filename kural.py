#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 23:16:36 2018

@author: abisek
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pygame
import pygame.camera
from pygame.locals import *
import os
from gtts import gTTS
#import pyttsx



def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

def predict_image(model,img_path):
    
    
    classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']    
    img_path= '/media/abisek/Important Files/R.K/++/Machine Learning/Projects/K/captures/'+img_path
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)  
    
    ans=pred[0]
   
    return classes[np.argmax(ans)]

def capture_images():
    
    DEVICE = '/dev/video0'
    SIZE = (640, 480)
    FILENAME = '/media/abisek/Important Files/R.K/++/Machine Learning/Projects/K/captures/capturetest'

    pygame.init()
    pygame.camera.init()
    display = pygame.display.set_mode(SIZE, 0)
    camera = pygame.camera.Camera(DEVICE, SIZE)
    camera.start()
    screen = pygame.surface.Surface(SIZE, 0, display)
    capture = True
    count=0
    while capture:
        screen = camera.get_image(screen)
        display.blit(screen, (0,0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == 113:
                capture = False
            elif event.type == KEYDOWN and event.key == 97:
                pygame.image.save(screen, FILENAME+str(count)+'.jpg')
                count+=1
    camera.stop()
    pygame.quit()

def make_predictions():

    model = load_model("projectKmodel2ver.h5")

    path='/media/abisek/Important Files/R.K/++/Machine Learning/Projects/K/captures'

    files= os.listdir(path)

    word=''

    for f in files:

        word+=predict_image(model,f)

    return word

def speak(mytext):

    engine = pyttsx.init();
    engine.say(mytext)
    engine.runAndWait() 
    
    


if __name__ == "__main__":

   #print(predict_image('test2.jpg'))
   #capture_images()

   

   text=make_predictions()

   print(text)

   #speak(text)

