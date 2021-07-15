import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
from keras.preprocessing import image

for filename in os.listdir('./archive/train/no'):
    print(filename)

    img = cv2.imread(os.path.join('./archive/train/no', filename))
    if img is None:
        print('err')
        continue