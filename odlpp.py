import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
from keras.preprocessing import image

for x in range(1, 10, 1):
    filename = ""
    filename = 'archive/test/yes/Y' + str(x) + '.jpg'
    print(filename)
    
    img = cv2.imread(filename)
    if img is None:
        continue

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    image = cv2.resize(new_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    image = image / 255.

    plt.imshow(image, cmap='gray')
    plt.show()