"""

@Author: Philip Yawen Guo

@ This repository is to automatic sitiching rota images according to

"""



import os
import glob
import numpy as np
import cv2
from Rota_stitching import fast_glcm
from numpy import float32
import matplotlib.pyplot as plt
from scipy.signal import correlate
from Rota_stitching.box_pairing import box_pairing

def skimage2opencv(src):
    src *= 255
    src.astype(int)
    cv2.cvtColor(src,cv2.COLOR_RGB2BGR)
    return src

def opencv2skimage(src):
    cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    src.astype(float32)
    src = src/255
    return src

path = "./examples2/"
img_list = glob.glob(path+"*.png")

if __name__ == '__main__':
    imgs = []
    mac_opt_vessel_imgs = []
    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        imgs.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img', img)

        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        contrast_enhanced_green_fundus = clahe.apply(img)
        r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
        R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
        f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
        f5 = clahe.apply(f4)

        # removing very small contours through area parameter noise removal
        ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
        mask = np.ones(f5.shape[:2], dtype="uint8") * 255
        contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) <= 5:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        im = cv2.bitwise_and(f5, f5, mask=mask)
        ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
        newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        mac_opt_vessel_imgs.append(newfin)

    ks = 50
    moving_coors=box_pairing(mac_opt_vessel_imgs[0],mac_opt_vessel_imgs[1],ks,10,3)

    for i in range(len(moving_coors)):
        moving_coors[i] = int(moving_coors[i])
    print(moving_coors)
    x1 = 300
    y1 = 300
    x2 = 300 + len(imgs[0])
    y2 = 300

    x_movestep = moving_coors[3] + (len(imgs[0][0])-moving_coors[1])
    y_movestep = moving_coors[0] - moving_coors[2]
    canvas = np.zeros((1000,1000,3),dtype=np.uint8)
    canvas[y1:y1+len(imgs[0]),x1:x1+len(imgs[0][0]),:] = imgs[0]
    canvas[y2+y_movestep:y2+len(imgs[1])+y_movestep,x2-x_movestep:x2-x_movestep+len(imgs[1][0]),:] = imgs[1]
    plt.imshow(canvas)
    plt.show()