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

def read_img(path):
    subjects_path = os.listdir(path)
    IDs = []
    for subject_path in subjects_path:
        name = subject_path[:subject_path.index("_")+8]
        if name not in IDs:
            IDs.append(name)

    image_paths = []
    for i in range(len(IDs)):
        each_pairs = [0,0,0,0,0]
        # L mac opt
        # R opt mac
        for subject in subjects_path:
            # image seq: rota_opt; raw_opt; rota_mac; raw_mac; name
            if "_L_" in subject:
                if IDs[i] in subject and "_L_Opt" in subject:
                    each_pairs[0] = subject + "/" + subject[:-3] + "RMapEnSumRNFL_3Resized.png"
                    each_pairs[1] = subject + "/" + "EnfaceAdjusted.png"
                if IDs[i] in subject and "_L_Mac" in subject:
                    each_pairs[2] = subject + "/" + subject[:-3] + "RMapEnSumRNFL_3Resized.png"
                    each_pairs[3] = subject + "/" + "EnfaceAdjusted.png"
                if IDs[i] in subject:
                    each_pairs[4] = IDs[i]+"_L_"
        image_paths.append(each_pairs)

        each_pairs = [0,0,0,0,0]
        for subject in subjects_path:
            # image seq: rota_mac; raw_mac; rota_opt; raw_opt; name
            if "_R_" in subject:
                if IDs[i] in subject and "_R_Mac" in subject:
                    each_pairs[0] = subject + "/" + subject[:-3] + "RMapEnSumRNFL_3Resized.png"
                    each_pairs[1] = subject + "/" + "EnfaceAdjusted.png"
                if IDs[i] in subject and "_R_Opt" in subject:
                    each_pairs[2] = subject + "/" + subject[:-3] + "RMapEnSumRNFL_3Resized.png"
                    each_pairs[3] = subject + "/" + "EnfaceAdjusted.png"
                if IDs[i] in subject:
                    each_pairs[4] = IDs[i]+"_R_"
        image_paths.append(each_pairs)
    return image_paths

path = "./Export GEN Extracted 20200513/"
image_paths = read_img(path)
if __name__ == '__main__':
    idx=0
    error_list = []
    for each_image_paths in image_paths:
        print("Current coping:",each_image_paths[4],"percentile:",idx/len(image_paths),"Index:",idx)
        idx+=1
        is_complete = True
        for element in each_image_paths:
            if element == 0:
                is_complete = False
        if is_complete:
            result_imgs = os.listdir("./result/")
            if each_image_paths[4]+"stitched.jpg" not in result_imgs:
                try:
                    imgs = []
                    mac_opt_vessel_imgs = []
                    img_list = [path+each_image_paths[1],path+each_image_paths[3]]
                    for i in range(len(img_list)):
                        img = cv2.imread(img_list[i])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

                    imgs = []
                    img1 = cv2.imread(path + each_image_paths[0])
                    img2 = cv2.imread(path + each_image_paths[2])
                    imgs = [img1,img2]
                    x1 = 100
                    y1 = 100
                    x2 = 100 + len(imgs[0])
                    y2 = 100

                    x_movestep = moving_coors[3] + (len(imgs[0][0])-moving_coors[1])
                    y_movestep = moving_coors[0] - moving_coors[2]
                    canvas = np.ones((800,600,3),dtype=np.uint8)*255
                    if "_L_" in each_image_paths[4]:
                        canvas[y1:y1+len(imgs[0]),x1:x1+len(imgs[0][0]),:] = imgs[0]
                        canvas[y2+y_movestep:y2+len(imgs[1])+y_movestep,x2-x_movestep:x2-x_movestep+len(imgs[1][0]),:] = imgs[1]
                    else:
                        canvas[y2+y_movestep:y2+len(imgs[1])+y_movestep,x2-x_movestep:x2-x_movestep+len(imgs[1][0]),:] = imgs[1]
                        canvas[y1:y1+len(imgs[0]),x1:x1+len(imgs[0][0]),:] = imgs[0]
                    cv2.imwrite("./result/"+each_image_paths[4]+"stitched.jpg",canvas)

                except:
                    print("error names", each_image_paths[4])
                    error_list.append(each_image_paths[4])

        else:
            error_list.append(each_image_paths[4])

np.save("error_result.npy",error_list)