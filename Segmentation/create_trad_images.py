import os
import cv2
from tqdm import tqdm


def resize_images(todataset=r'../dataset/traditional/Fluo-N2DH-GOWT1', fromdataset=r'../Fluo-N2DH-GOWT1'):
    fromSet = []

    for num in os.listdir(fromdataset):
        if '_' not in num:
            fromSet.append(num)

    toSet = os.listdir(todataset)

    for fromNum in fromSet:
        for toNum in toSet:
            if fromNum == toNum:
                fromfilepath = os.path.join(fromdataset, fromNum)
                tofilepath = os.path.join(todataset, toNum)

                for filename in tqdm(os.listdir(fromfilepath)):
                    imgpath = os.path.join(fromfilepath,filename)
                    if cv2.imread(imgpath) is not None:
                        img = cv2.imread(imgpath, -1)
                        resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(tofilepath,filename), resized)


def resize_masks(todataset=r'../dataset/traditional/Fluo-N2DH-GOWT1', fromdataset=r'../Fluo-N2DH-GOWT1'):
    fromset = []

    for num in os.listdir(fromdataset):
        if '_' in num:
            fromset.append(num)

    toset = os.listdir(todataset)

    for fromNum in fromset:
        for toNum in toset:
            if fromNum == toNum:
                fromfilepath = os.path.join(fromdataset, fromNum)
                tofilepath = os.path.join(todataset, toNum)

                fromfolders = os.listdir(fromfilepath)
                tofolders = os.listdir(tofilepath)

                for fromfolder in fromfolders:
                    for tofolder in tofolders:
                        if fromfolder == tofolder:
                            for filename in tqdm(os.listdir(os.path.join(fromfilepath, fromfolder))):
                                if '.tif' not in filename:
                                    continue
                                imgpath = os.path.join(fromfilepath, fromfolder, filename)
                                if cv2.imread(imgpath) is not None:
                                    img = cv2.imread(imgpath, -1)
                                    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                                    cv2.imwrite(os.path.join(tofilepath, tofolder, filename), resized)


resize_images()
resize_masks()