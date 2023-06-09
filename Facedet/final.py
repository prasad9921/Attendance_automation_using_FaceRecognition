#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:54:12 2020

@author: basu
"""


import cv2
import numpy as np
import os
import pandas as pd
from final2 import multi_occ_remover 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')   #load trained model
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter, the number of persons you want to include
id = 2 #two persons (e.g. Jacob, Jack)

#################################################################
"""here lies the code bhaswanth added"""
lis = []  #created for adding a list of list of names
#i.e lis is a 2d list consisting of list of names in every given index. its basically a list of lists

##############################################################

names = ['','saikiran','chaari']  #key in names, start from the second place, leave first empty

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    name_adder = []
    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
         
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence > 40):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        ##############################################################
        if(id != "unknown"):
            name_adder.append(str(id))
            
        ###############################################################    
    lis.append(name_adder)
    i=0
    k=0
    k=0
    for k in range(len(names)):
        count=0
        for i in range(len(lis)):
            for j in range(len(lis[i])):
                if(lis[i][j]==names[k]):
                    count = count+1
                    if(count==1):
                        pass
                    else:
                        lis[i][j]=""
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        gh=pd.DataFrame(lis)
        gh.to_excel("hello.xlsx")
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
