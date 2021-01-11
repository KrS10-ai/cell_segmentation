import os
import cv2
from tqdm import tqdm


def resize_images(todataset=r'../dataset/unet/Fluo-N2DH-GOWT1/Training', fromdataset=r'../Fluo-N2DH-GOWT1'):
    fromset = []

    for num in os.listdir(fromdataset):
        if '_' not in num:
            fromset.append(num)

    toset = os.listdir(todataset)

    for fromNum in fromset:
        for toNum in toset:
            if fromNum == toNum:
                fromfilepath = os.path.join(fromdataset, fromNum)
                tofilepath = os.path.join(todataset, toNum)

                for filename in tqdm(os.listdir(fromfilepath)):
                    imgpath = os.path.join(fromfilepath,filename)
                    if cv2.imread(imgpath) is not None:
                        img = cv2.imread(imgpath, -1)
                        resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(tofilepath,filename), resized)


def resize_masks(todataset=r'../dataset/unet/Fluo-N2DH-GOWT1/Training', fromdataset=r'../Fluo-N2DH-GOWT1'):

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

                # The SEG and TRA folders
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


def binary_masks(todataset=r'../dataset/unet/Fluo-N2DH-GOWT1/Training', fromdataset=r'../Fluo-N2DH-GOWT1'):
    fromset = []

    for num in os.listdir(fromdataset):
        if '_' in num:
            fromset.append(num)

    toset = []
    for num in os.listdir(todataset):
        if 'TMSK' in num:
            toset.append(num)

    for fromNum in fromset:
        for toNum in toset:
            if fromNum in toNum:
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
                                    ret, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY)
                                    cv2.imwrite(os.path.join(tofilepath, tofolder, filename), thresh)


def set_segtrain_images(testfolder=r'../dataset/unet/Fluo-N2DH-GOWT1/Training'):
    image_number_set = []

    for num in os.listdir(testfolder):
        if '_' not in num:
            image_number_set.append(num)

    mask_number_set = []
    for num in os.listdir(testfolder):
        if '_GT' in num:
            mask_number_set.append(num)

    for imagenum in image_number_set:
        for masknum in mask_number_set:
            if imagenum in masknum:
                imagesfilepath = os.path.join(testfolder, imagenum)
                masksfilepath = os.path.join(testfolder, masknum)

                masktypes = os.listdir(masksfilepath)

                for type in masktypes:
                    if type == 'SEG':
                        masks_num = []

                        # removing the images with ground truth masks
                        for maskfilename in tqdm(os.listdir(os.path.join(masksfilepath, type))):
                            for number in range(len(os.listdir(imagesfilepath))):
                                numstr = str(number).zfill(3)
                                if numstr in maskfilename:
                                    masks_num.append(numstr)

                        for img_num in masks_num:
                            imagename = 't'+img_num+'.tif'

                            if os.path.exists(os.path.join(imagesfilepath, imagename)):
                                # print(os.path.join(imagesfilepath, imagename))
                                os.remove(os.path.join(imagesfilepath, imagename))


def set_segtrain_binmask(trainfolder=r'../dataset/unet/Fluo-N2DH-GOWT1/Training'):
    bin_number_set = []

    for num in os.listdir(trainfolder):
        if 'TMSK' in num:
            bin_number_set.append(num)

    mask_number_set = []
    for num in os.listdir(trainfolder):
        if '_GT' in num:
            mask_number_set.append(num)

    for binnum in bin_number_set:
        for masknum in mask_number_set:
            if binnum[1] in masknum[1]:
                for bintype in os.listdir(os.path.join(trainfolder, binnum)):
                    for masktype in os.listdir(os.path.join(trainfolder, masknum)):
                        if bintype == 'SEG' and bintype==masktype:
                            masksfilepath = os.listdir(os.path.join(trainfolder, masknum, masktype))

                            for mask in masksfilepath:
                                if os.path.exists(os.path.join(trainfolder, binnum, bintype, mask)):
                                    os.remove(os.path.join(trainfolder, binnum, bintype, mask))


resize_images()
resize_masks()
binary_masks()

set_segtrain_images()
set_segtrain_binmask()
