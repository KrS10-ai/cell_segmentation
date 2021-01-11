import numpy as np
import cv2
from tqdm import tqdm
import random
from matplotlib import pyplot as plt


image = cv2.imread(r'../dataset/unet/Fluo-N2DH-GOWT1/Testing/01/t008.tif', 0)
cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\unet/image.png', image)

# binary mask
binary = cv2.imread('../dataset/unet/Fluo-N2DH-GOWT1/Testing/drive/01_RES/mask008.tif', 0)
cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\unet/binary.png', binary)

# track cells
contours2, hierarchy2 = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
track = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(track, contours2, -1, (0, 0, 255), 1)
cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\unet/segmented.png', track)

# create marker
marker = np.zeros(binary.shape, dtype=np.uint16)
for i in tqdm(range(len(contours2))):
    cv2.fillPoly(marker, pts=[contours2[i]], color=i)

plt.imshow(marker)
plt.imsave(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\unet/marker.png', marker)

# plt.hist(equal.ravel(), 255,[0, 255]);
plt.show()

