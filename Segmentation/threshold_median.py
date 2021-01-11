import os
import cv2
import numpy as np
from tqdm import tqdm


def threshold_images(todataset=r'../dataset/traditional/Fluo-N2DH-GOWT1', fromdataset=r'../Fluo-N2DH-GOWT1'):
    fromset = []

    for num in os.listdir(fromdataset):
        if '_' not in num:
            fromset.append(num)

    toset = []
    for num in os.listdir(todataset):
        if '_RES' in num:
            toset.append(num)

    for fromNum in fromset:
        for toNum in toset:
            if fromNum in toNum:
                fromfilepath = os.path.join(fromdataset, fromNum)
                tofilepath = os.path.join(todataset, toNum)

                counter = 0                       # count the files
                for filename in tqdm(os.listdir(fromfilepath)):
                    imgpath = os.path.join(fromfilepath, filename)
                    if cv2.imread(imgpath) is not None:
                        img = cv2.imread(imgpath, -1)
                        resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

                        equal = cv2.equalizeHist(resized)
                        blur = cv2.medianBlur(equal, 5)

                        ret, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)

                        # non-cell removal
                        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        new = np.zeros(thresh.shape, dtype=np.uint8)
                        for i in range(len(contours)):
                            if cv2.contourArea(contours[i]) > 100:
                                cv2.fillPoly(new, pts=[contours[i]], color=255)

                        kernel = np.ones((3, 3), np.uint8)
                        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                        # separate cells touching
                        erode = cv2.erode(new, kernel2, iterations=5)

                        # create binary mask
                        binary = cv2.dilate(erode, kernel2, iterations=5)

                        # create marker
                        contours2, hierarchy2 = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        marker = np.zeros(binary.shape, dtype=np.uint16)
                        for i in tqdm(range(len(contours2))):
                            cv2.fillPoly(marker, pts=[contours2[i]], color=i)

                        maskfilename = 'mask'+str(counter).zfill(3)+'.tif'
                        cv2.imwrite(os.path.join(tofilepath, maskfilename), marker)
                        counter += 1


threshold_images()
