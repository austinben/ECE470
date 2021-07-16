import numpy as np
import os
import cv2
import imutils
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt

def crop_image(img):

    #convert the images to greyscale and add a slight guassian blur
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    #threshold the image and perform erosions and dilations to remove noise
    thresh = cv2.threshold(gray,45,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    #detect contours in the threshold image, then get the largest one we can find.
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    #map extreme points of the image.ext
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original
    new_image = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return new_image      