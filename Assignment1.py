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
    
    return img

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

#connected component labeling
def connectedComponentLabeling(image, thresh) :
    #convert the image to binary using a threshold
    binary_image = (image > thresh).astype(np.uint8) ##idk if I can do this this way???

    labels = np.zeros_like(binary_image)
    label = 1  #start labeling from 1
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i][j] == 1 and labels[i][j] == 0:
                dfs(binary_image, labels, i, j, label)
                label += 1

    return labels

#search for connected pixels
def dfs(binary_image, labels, i, j, label):
    stack = [(i, j)]
    while stack:
        i, j = stack.pop()

        if 0 <= i < binary_image.shape[0] and 0 <= j < binary_image.shape[1]:
            if binary_image[i][j] == 1 and labels[i][j] == 0:
                labels[i][j] = label

                for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    stack.append((i + x, j + y))


#classify oring as a pass or fail
def classifyOring(labels):
    #check if there are any labels
    if np.max(labels) == 0:
        return "Fail"

    #check if there are more than 1 label
    if np.max(labels) > 1:
        return "Fail"

    #check if the label is a circle
    if np.max(labels) == 1:
        return "Pass"

    return "Fail"

                
#read in an image into memory
for i in range(1,16):
    img = cv.imread('Orings/Oring' + str(i) +'.jpg',0)
    hist = imhist(img)

    #finding threshold, and add it to image
    thresh = find_peak(hist)-70
    print(thresh)
    cv.imshow('original image',img)
    thresh_img = threshold(img,thresh)
    img = cv.cvtColor(thresh_img,cv.COLOR_GRAY2BGR)
    cv.putText(img,str(thresh),(10,30),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),thickness=2)
    cv.imshow('thresholded image',img)

    #binary morphology called and displayed
    binImage = binaryMorphology(thresh_img)
    cv.imshow("bin Image", binImage)

    # Perform connected component labeling
    labels = connectedComponentLabeling(binImage, thresh)

    #check if connected component labeling returned null
    if labels is None:
        print("Error: Connected component labeling failed.")
    else:
        cv.imshow('Labeled Image', labels.astype(np.uint8) * 100) # *100 controls visiblity for the labels, can be set to anything


    #classify the oring
    classification = classifyOring(labels)
    print("Oring " + str(i) + " is a " + classification)

    cv.waitKey()
    plt.plot(hist)
    plt.show()
    
cv.destroyAllWindows()