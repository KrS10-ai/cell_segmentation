import os
import cv2
import numpy as np
from tqdm import tqdm


for imgpath in tqdm(os.listdir('../dataset/unet/Fluo-N2DH-GOWT1/Testing/drive/02_RES')):
    if cv2.imread(os.path.join('../dataset/unet/Fluo-N2DH-GOWT1/Testing/drive/02_RES', imgpath)) is not None:
        binary = cv2.imread(os.path.join('../dataset/unet/Fluo-N2DH-GOWT1/Testing/drive/02_RES', imgpath), 0)

        # create marker
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        marker = np.zeros(binary.shape, dtype=np.uint16)
        for i in tqdm(range(len(contours))):
            cv2.fillPoly(marker, pts=[contours[i]], color=i)

        cv2.imwrite(os.path.join('../dataset/unet/Fluo-N2DH-GOWT1/Testing/02_RES', imgpath), marker)
