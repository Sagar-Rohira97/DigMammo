
#Following is the code for image processing of a digital mammogram

import numpy as np
import cv2
from matplotlib import pyplot as plt

#Code to read an image so that it can be processed
img = cv2.imread('bcancer4.jpg', 0)


#Code to flip the image so that I can easily
#remove the mucle component after segmantation
if img.shape[0]>img.shape[1]: img=cv2.flip(img,1,img)
z1 = img.reshape((-1,1))
z1 = np.float32(z1)


#Segmentation using K Means Clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(z1,5,None,criteria,10,flags)
centers = np.uint8(centers)
res = centers[labels.flatten()]
output = res.reshape((img.shape))

#Thresholding
ret,thresh = cv2.threshold(output , centers[2] , 255, cv2.THRESH_BINARY_INV)

#Edge Detection
canny = cv2.Canny(thresh,10,100)
canny = cv2.dilate(canny, None,iterations = 1)
canny = cv2.erode(canny, None, iterations = 1)
ret2,thresh2 = cv2.threshold(canny,0,255,cv2.THRESH_BINARY)

#Cropping
h = img.shape[0]
w = img.shape[1]
thresh3 = thresh2[0:h,int(centers[2]-30): w]
cnts , heirarchy = cv2.findContours(thresh3 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

#Classification based on the area of each contour
for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area>=3000 and area<=35000: print("Stage1 cancer")
    elif area>35000 and area<=85000 : print("Stage2 cancer")
    elif area>85000 and area<=250000 : print("Stage3 cancer")
    else: print("Not malignant cancer")



#Superimposition of the boundaries on the original image
added_image = cv2.addWeighted(img,0.7,thresh2,0.5,0)


#Results showing the images at every stage
cv2.imshow('img' , img)
cv2.imshow('KMeans' , output)
cv2.imshow('Thresholded',thresh)
cv2.imshow('canny' , canny)
cv2.imshow('Cropped', thresh3)
cv2.imshow('Superimposed', added_image)

#Showing all the images together
plt.subplot(2,3,1),plt.imshow(img , 'gray')
plt.subplot(2,3,2),plt.imshow(output , 'gray')
plt.subplot(2,3,3),plt.imshow(thresh , 'gray')
plt.subplot(2,3,4),plt.imshow(canny, 'gray')
plt.subplot(2,3,5),plt.imshow(thresh2, 'gray')
plt.subplot(2,3,6),plt.imshow(added_image, 'gray')
plt.show()


