import os
import cv2
import numpy as np
from tqdm import tqdm


for imgpath in tqdm(os.listdir('../dataset/traditional/Fluo-N2DH-GOWT1/02')):
    if cv2.imread(os.path.join('../dataset/traditional/Fluo-N2DH-GOWT1/02', imgpath)) is not None:
        image = cv2.imread(os.path.join('../dataset/traditional/Fluo-N2DH-GOWT1/02', imgpath), 0)
        imageResized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)


        equal = cv2.equalizeHist(imageResized)
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

        # remove noise
        # opening = cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel, iterations=2)

        # foreground
        erode = cv2.erode(new, kernel2, iterations=5)

        # binary mask
        binary = cv2.dilate(erode, kernel2, iterations=5)


        # track cells
        contours2, hierarchy2 = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        track = cv2.cvtColor(imageResized, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(track, contours2, -1, (0, 0, 255), 1)
        cv2.imwrite(
            os.path.join(r'../dataset/traditional/Fluo-N2DH-GOWT1/02_Segmented', imgpath),
            track)

        # create marker
        marker = np.zeros(binary.shape, dtype=np.uint16)
        for i in tqdm(range(len(contours2))):
            cv2.fillPoly(marker, pts=[contours2[i]], color=i)

        segmentation = cv2.hconcat(
            [cv2.cvtColor(imageResized, cv2.COLOR_GRAY2BGR), cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
             track])
