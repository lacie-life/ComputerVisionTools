# Python program to illustrate
# foreground extraction using
# GrabCut algorithm

# organize imports
import numpy as np
import cv2
from matplotlib import pyplot as plt

# path to input image specified and
# image is loaded with imread command
image = cv2.imread('E:\\Github\\ComputerVisionTools\\scritps\\imgs\\test.png')

# create a simple mask image similar
# to the loaded image, with the
# shape and return type
mask = np.zeros(image.shape[:2], np.uint8)

# specify the background and foreground model
# using numpy the array is constructed of 1 row
# and 65 columns, and all array elements are 0
# Data type for the array is np.float64 (default)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

# define the Region of Interest (ROI)
# as the coordinates of the rectangle
# where the values are entered as
# (startingPoint_x, startingPoint_y, width, height)
# these coordinates are according to the input image
# it may vary for different images
rectangle = (100, 330, 380, 370)
rectangle1 = (480, 240, 400, 300)
rectangle2 = (970, 70, 300, 320)

# apply the grabcut algorithm with appropriate
# values as parameters, number of iterations = 3
# cv2.GC_INIT_WITH_RECT is used because
# of the rectangle mode is used
cv2.grabCut(image, mask, rectangle1,
            backgroundModel, foregroundModel,
            3, cv2.GC_INIT_WITH_RECT)

# cv2.grabCut(image, mask, rectangle1,
#             backgroundModel, foregroundModel,
#             3, cv2.GC_INIT_WITH_RECT)
#
# cv2.grabCut(image, mask, rectangle2,
#             backgroundModel, foregroundModel,
#             3, cv2.GC_INIT_WITH_RECT)

# In the new mask image, pixels will
# be marked with four flags
# four flags denote the background / foreground
# mask is changed, all the 0 and 2 pixels
# are converted to the background
# mask is changed, all the 1 and 3 pixels
# are now the part of the foreground
# the return type is also mentioned,
# this gives us the final mask
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# The final mask is multiplied with
# the input image to give the segmented image.
image = image * mask2[:, :, np.newaxis]

ret, bw_img = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

# output segmented image with colorbar
plt.imshow(bw_img)
plt.savefig('test2.svg', format='svg', dpi=1200)
# plt.colorbar()
plt.show()





