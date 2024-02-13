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

#binary morphology
def binaryMorphology(img):
    kernel = np.ones((3, 3), np.uint8) #structuring element using square

    #closing image
    dilated_img = dilate(img, kernel)
    closed_img = erode(dilated_img, kernel)

    return closed_img


#dilate img
def dilate(img, kernel):
    dilated_img = np.zeros_like(img)
    for y in range(1, img.shape[0] - 1):

        for x in range(1, img.shape[1] - 1):
            #where at least one element has to be true
            if np.any(img[y - 1:y + 2, x - 1:x + 2] == 255):
                dilated_img[y, x] = 255

    return dilated_img

#erode img
def erode(img, kernel):
    eroded_img = np.zeros_like(img)
    for y in range(1, img.shape[0] - 1):

        for x in range(1, img.shape[1] - 1):
            #all elements have to be true
            if np.all(img[y - 1:y + 2, x - 1:x + 2] == 255):
                eroded_img[y, x] = 255

    return eroded_img


# Connected Component Labelling - Algorithm
def connected_component_labelling(img):
    curlab = 1
    goTo = 2

    height, width = img.shape[:2]  # Assuming img is a 2D array

    # Define a queue for pixels to be processed
    queue = []

    # Function to check if a pixel is within image bounds and is a foreground pixel
    def is_foreground_pixel(y, x):
        return 0 <= y < height and 0 <= x < width and img[y, x] == 255

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # If the pixel is a foreground pixel and not already labelled
            #if is_foreground_pixel(y, x):
                #img[y, x] = curlab  # Label the pixel with curlab
                #queue.append((y, x))  # Add pixel coordinates to the queue

                # Process pixels in the queue
                

                 curlab += 1  # Increment curlab for the next connected component

    return curlab


                
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

    binImage = binaryMorphology(img)
    cv.imshow("bin Image", binImage)


    #connected compound labeling
    connected_component_labelling(binImage)

    cv.waitKey()
    plt.plot(hist)
    plt.show()
    
cv.destroyAllWindows()