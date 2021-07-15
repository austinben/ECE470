import numpy as np
import os
import cv2
import imutils
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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

    # save_filename = 'processed/train/no/N' + str(x) + '.jpg'
    # x +=1
    # cv2.imwrite(save_filename, im1)

# def prepare_data(directories, img_size):
#     IMAGES_X = [] 
#     IMAGES_Y = []
#     width = img_size
#     height = img_size

#     for direc in directories:
#         for filename in os.listdir(direc):
#             img = cv2.imread(direc + '//' + filename)
#             if img is None:
#                 print(err)
#                 exit(0)
#             img = crop_image(img) #send to our method to crop
#             img = cv2.resize(img, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
#             img = img/255 #normalize
#             IMAGES_X.append(img) #append to array in a numpy array style
#             if direc[-3:] == 'yes':
#                 IMAGES_Y.append([1])
#             else:
#                 IMAGES_Y.append([0])

#     IMAGES_X = np.array(IMAGES_X)
#     IMAGES_Y = np.array(IMAGES_Y)

#     # #shuffle
#     # IMAGES_X, IMAGES_Y = shuffle(IMAGES_X, IMAGES_Y)

#     return IMAGES_X, IMAGES_Y


# def data_split(IMAGES_X, IMAGES_Y):
#     TRAIN_X, X_TEST_VAL, TRAIN_Y, Y_TEST_VAL = train_test_split(IMAGES_X, IMAGES_Y, test_size=0.2)
#     X_TEST, X_VAL, Y_TEST, Y_VAL = train_test_split(X_TEST_VAL, Y_TEST_VAL, test_size=0.5)

#     return TRAIN_X, TRAIN_Y, X_VAL, Y_VAL, X_TEST, Y_TEST 


