import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt


#implement thresholding ourselves using loops (soooo slow in python)
def threshold(img,thresh):
    for y in range(0, img.shape[0]):#loop through rows
        for x in range(0, img.shape[1]):#loop through columns
            if img[y,x] < thresh:
                img[y,x] = 255
            else:
                img[y,x] = 0

def imhist(img):
    hist = np.zeros(256)
    for y in range(0, img.shape[0]):#loop through rows
        for x in range(0, img.shape[1]):#loop through columns
            hist[img[y,x]]+=1
    return hist

def find_peak(hist):
    peak_count = 0
    peak_index = 0
    for i in range(hist.size):
        if hist[i] > peak_count:
            peak_count = hist[i]
            peak_index = i
    return peak_index

#def binaryMorphology(img):
    

                
#read in an image into memory
for i in range(1,16):
    img = cv.imread('Orings/Oring' + str(i) +'.jpg',0)
    hist = imhist(img)
    #thresh = np.argmax(hist)-70
    thresh = find_peak(hist)-70
    print(thresh)
    cv.imshow('original image',img)
    #thresh = 100
    threshold(img,thresh)
    img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    cv.putText(img,str(thresh),(10,30),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),thickness=2)
    cv.imshow('thresholded image',img)
    cv.waitKey()
    plt.plot(hist)
    plt.show()
    
cv.destroyAllWindows()