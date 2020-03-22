import cv2,os
import numpy as np
from PIL import Image




#%% Function Section       
    
    
def Save_LabelIDs(fn, labelNames):
    with open(fn, 'w') as outf:
        for s in labelNames:
          outf.write('{}\n'.format(s)) # writes current line in opened fn file
       

def NamingByAssignedName(inputPath, outputPath, outputFileName=""):
    #get the path of all the files in the folder
    imageFiles=[os.path.join(inputPath,f) for f in os.listdir(inputPath)]     
    outputFiles = [os.path.join(outputPath,f) for f in os.listdir(outputPath)] 
    
    fileId = len(outputFiles)
    
    #now looping through all the images and greyscale them and cut out faces
    for imageFile in imageFiles:
        
        if outputFileName == "":
          origFileName = os.path.split(imageFile)[-1]
        else:
          origFileName = "{}{}.jpg".format(outputFileName, fileId)
          fileId+=1
          
        #loading the image and converting it to gray scale
        #pilImage=Image.open(imageFile).convert('L')
        # Load an color image in grayscale
        img = cv2.imread(imageFile,0)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.imwrite(outputPath+origFileName,img[y:y+h,x:x+w])
        
    

def Delete_Files(sDir, delFileExts=[], deleteSubDir=False):
    if(sDir == '//' or sDir == "\\"): return
    else:
        for root, dirs, files in os.walk(sDir, topdown=False):
            for name in files:
                if len(delFileExts)<=0:
                    os.remove(os.path.join(root, name))
                else:
                    for sExt in delFileExts:
                      if name.endswith(sExt): #(".jpg") 
                        os.remove(os.path.join(root, name))
            if deleteSubDir:
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                    
                    
#%%
                    
                    
recognizer = cv2.face.LBPHFaceRecognizer_create()

inputPath1 = 'dataSet\\'
outputPath = 'Face\\'


if not os.path.exists(outputPath):
    os.makedirs(outputPath)
else:
    Delete_Files(outputPath, ['.jpg'])
    

#fnClassfier = 'lbpcascade_frontalface.xml'
fnClassfier = 'haarcascade_frontalface_default.xml'
#fnClassfier = 'haarcascade_frontalface_alt.xml'

#face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(fnClassfier)

#NamingByAssignedName(inputPath1, outputPath, "")
labelNames = []
labelId=0
for root, dirs, files in os.walk(inputPath1, topdown=False):
   for name in dirs:
       labelId+=1
       labelNames.append("{}={}".format(labelId,name))
       path = os.path.join(inputPath1, name)
       NamingByAssignedName(path, outputPath, "{}.{}.".format(name, labelId))
       
Save_LabelIDs('labelIDs.txt', labelNames)
        
cv2.destroyAllWindows()

print("\nImages prepared and saved in : \n\t\"{}\"".format(os.path.abspath(outputPath)))