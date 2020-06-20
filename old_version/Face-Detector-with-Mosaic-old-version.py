import numpy as np
import pylab
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import random

FilePath = '/home/will/Videos/Face-recognition-demo/Processing'
face_cascade = cv2.CascadeClassifier('/home/will/Real-Time-Face-Detection/haarcascade_frontalface_default.xml')

for file in os.listdir(FilePath):
    imgPath = os.path.join(FilePath,file)
    img = plt.imread(imgPath)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    print(faces)
    if faces == []:
        print(imgPath)
    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # draw rectangle around Detected Face
        #print(x,y,w,h)
        head = img[y:y+h, x:x+w]

        head2 = head[::10, ::10]
        #plt.imshow(head2)
        #pylab.show()
        #plt.imshow(head2)

        img2 = img.copy()

        for i in range(h//10):
            for j in range(w//10):
                img2[y + i * 10:y+10 + i * 10, x + j * 10:x+10 + j * 10] = head2[i][j]
        im2 = Image.fromarray(img2.astype("uint8"))
        im2.show()
