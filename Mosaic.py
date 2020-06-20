import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import argparse

#FilePath = '/home/will/Videos/Face-recognition-demo/Processing'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filepath", required=False,
                default= '/home/will/Videos/Face-recognition-demo/Processing',
                help="image file")
ap.add_argument("-p", "--prototxt", required=False,
                default= '/home/will/FaceDetection_Realtime/deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,
                default= '/home/will/FaceDetection_Realtime/res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--save-face", type = str, required=False,
                default='T', choices=['F', 'T'], help="detect face and save face")
ap.add_argument("-i", "--img-save", type = str, required=False,
                default= './save_image/',
                help="the directory for saving image")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

for file in os.listdir(args['filepath']):
    name, extension = file.split('.')
    imgPath = os.path.join(args['filepath'],file)
    img = plt.imread(imgPath)

    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #print(x,y,w,h)
        head = img[startY:endY, startX:endX]

        # [a,b] a代表垂直方向坐标，b代表水平方向坐标
        head2 = head[::10, ::10]

        img2 = img.copy()

        for i in range((endY-startY)//10):
            for j in range((endX-startX)//10):
                img2[startY + i * 10:startY+10 + i * 10, startX + j * 10:startX+10 + j * 10] = head2[i][j]
        #im2 = Image.fromarray(img2.astype("uint8"))
        #im2.show()
        if args["save_face"] == 'T':
            os.makedirs(args["img_save"], exist_ok=True)
            img3 = img2[:, :, [2, 1, 0]]
            cv2.imwrite(args["img_save"] + str(name) + '.jpg', img3)

