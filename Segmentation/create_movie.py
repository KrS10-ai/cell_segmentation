import os
import numpy as np
import cv2


def create_mov(folder, videoname='video'):
    images = []

    for filename in os.listdir(folder):
        imgpath = os.path.join(folder, filename)
        image = cv2.imread(imgpath)

        # size is (width, height)
        size = (image.shape[1], image.shape[0])
        if image is not None:
            images.append(image)

    out = cv2.VideoWriter('../'+videoname+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(images)):
        out.write(images[i])
    out.release()


#create_mov(r'../dataset/Fluo-N2DH-GOWT1/Training/01_Images/', 'image')
#create_mov(r'../dataset/Fluo-N2DH-GOWT1/Training/01_Mask/', 'mask')
create_mov(r'../dataset/traditional/Fluo-N2DH-GOWT1/01_Segmented/', 'segment')

#create_mov(r'../dataset/Fluo-N2DH-GOWT1/Training/02_Images/', 'image2')
#create_mov(r'../dataset/Fluo-N2DH-GOWT1/Training/02_Mask/', 'mask2')
create_mov(r'../dataset/traditional/Fluo-N2DH-GOWT1/02_Segmented/', 'segment2')