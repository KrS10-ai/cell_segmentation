import numpy as np
import cv2
from tqdm import tqdm
import random
from matplotlib import pyplot as plt


image = cv2.imread(r'../Fluo-N2DH-GOWT1/01/t014.tif', 0)
imageResized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/image.png', imageResized)

equal = cv2.equalizeHist(imageResized)
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/equal.png', equal)

blur = cv2.medianBlur(equal, 5)
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/median.png', blur)

ret, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/threshold.png', thresh)

# non-cell removal
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

new = np.zeros(thresh.shape, dtype=np.uint8)
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 100:
        cv2.fillPoly(new, pts=[contours[i]], color=255)

#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/removefrag.png', new)

kernel = np.ones((3, 3), np.uint8)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# remove noise
# opening = cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel, iterations=2)

# foreground
erode = cv2.erode(new, kernel2, iterations=5)
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/erosion.png', erode)

# binary mask
binary = cv2.dilate(erode, kernel2, iterations=5)
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/binary.png', binary)


# track cells
contours2, hierarchy2 = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
track = cv2.cvtColor(imageResized, cv2.COLOR_GRAY2BGR)
cv2.drawContours(track, contours2, -1, (0, 0, 255), 1)
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/segmented.png', track)

# create marker
marker = np.zeros(binary.shape, dtype=np.uint16)
for i in tqdm(range(len(contours2))):
    cv2.fillPoly(marker, pts=[contours2[i]], color=i)

segmentation = cv2.hconcat([cv2.cvtColor(imageResized, cv2.COLOR_GRAY2BGR), cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
                            track])

cv2.imshow('thresh', thresh)
#plt.imshow(marker)
#plt.imsave(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/marker.png', marker)

#plt.hist(equal.ravel(), 255,[0, 255])
#cv2.imwrite(r'C:\Users\Student\Desktop\2020_Study\COMP700_Project\COMP700_Report\My_Report\results\traditional/higher.png', thresh)
#plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

