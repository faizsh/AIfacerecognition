

#%%
# Standard library----------------------------------------------
import sys
sys.path.append('../data')
sys.path.append('D:\SourceCode\Python\DanielSourceCode\DeepLearning\RvLib')
import os
import time
from datetime import datetime

import cv2
from skimage.filters import threshold_mean

# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

        
#%% Function Section

def Draw_Text(img, sTxt, aX=30, aY=30):
    if ""==sTxt: return
    cv2.putText(image, str(sTxt) ,(aX,aY), font, 
        fntSize,(0,255,255), fntThickness,cv2.LINE_AA)
    

def CvBGR_To_RGB(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


def Load_LabelIDs(fn):
    labelNames = []
    with open(fn) as f:
        data = f.readlines()
        for i in range(len(data)):
            lines = str(data[i]).split("\\n")
            for s in lines:
              labelNames.append(s)
    return labelNames

labelNames = Load_LabelIDs('labelIDs.txt')
labelDics = {}
for s in labelNames:
    strs = str(s).split("=")
    labelDics[strs[0]] = strs[1].split("\n")[0]
#%% Constant / Variables
    
    
font = cv2.FONT_HERSHEY_SIMPLEX
fntSize = 1
fntThickness = 1

            
#fnClassfier = 'lbpcascade_frontalface.xml'
fnClassfier = 'haarcascade_frontalface_default.xml'
#fnClassfier = 'haarcascade_frontalface_alt.xml'

faceCascade = cv2.CascadeClassifier(fnClassfier)

fname = "trainner.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)

camera = cv2.VideoCapture(0)
while True:
    return_value,image = camera.read()  
    imgInfo = np.asarray(image).shape     
    if len(imgInfo)<2: break
    imgH=imgInfo[0]
    imgW=imgInfo[1]
    imgChannel=imgInfo[2] 
    
    #cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
         cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
         Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
         #print (conf)
         key = "{}".format(Id)
         if (key in labelDics):  
             Id = labelDics[key]
         else:
             Id='Unknown'
         newimg = cv2.putText(image, str(Id), (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
#         newName = origFileName+ "-" + Id + str(increment)
#         print (newName)
#         print ("------")
#         cv2.imwrite(outputPath+ newName +'.jpg',newimg)
         
    
    try:                 
        Draw_Text(image, "esc:exit")
        cv2.imshow('image',image)
        
#        plt.figure(figsize=(10,5))
#        plt.imshow(CvBGR_To_RGB(image)); 
        #plt.imshow( cv2.cvtColor(image,cv2.COLOR_BGR2RGB) )
            
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  #esc   ord('s'):
            #cv2.imwrite('test.jpg',image)
            break
            
    except ValueError:
        break

camera.release()
cv2.destroyAllWindows()
