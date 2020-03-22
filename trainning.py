
import cv2,os
import numpy as np
from PIL import Image



#%% Function Section       
   
def ForceDir(path):
    if not os.path.isdir(path):
        os.mkdir(path) 
        
 
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imageFiles=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imageFile in imageFiles:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imageFile).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imageFile)[-1].split('.')[1])
        faceSamples.append(imageNp)
        Ids.append(Id)
    return faceSamples, np.array(Ids)       


        
#%%
    
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()

path = 'Face\\'
fnTrain = 'trainner.yml'



faces, Ids = getImagesAndLabels(path)
recognizer.train(faces,Ids)
recognizer.write(fnTrain)
cv2.destroyAllWindows()

if len(faces)>0:
   print("\nTraining data saved as : \"{}\"".format(fnTrain))

