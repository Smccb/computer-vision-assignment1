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

# binary morphology
def binaryMorphology(img):
    # closing image
    dilated_img = dilate(img)
    closed_img = erode(dilated_img)

    return closed_img

# dilate img
def dilate(img):
    dilated_img = np.zeros_like(img)
    for y in range(1, img.shape[0] - 1):

        for x in range(1, img.shape[1] - 1):
            # where at least one element has to be true
            if np.any(img[y - 1:y + 2, x - 1:x + 2] == 255):
                dilated_img[y, x] = 255

    return dilated_img

# erode img
def erode(img):
    eroded_img = np.zeros_like(img)
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            # all elements have to be true
            if np.all(img[y - 1:y + 2, x - 1:x + 2] == 255):
                eroded_img[y, x] = 255

    return eroded_img

# connected component labeling - uses lecture 2 slide 32
def connectedComponentLabeling(image, thresh) :
    # convert the image to binary using a threshold
    binary_image = (image > thresh).astype(np.uint8)

    labels = np.zeros_like(binary_image)
    curLab = 1  # start labeling from 1
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i][j] == 1 and labels[i][j] == 0:
                labels = dfs(binary_image, labels, i, j, curLab)
                curLab += 1

    return labels

# check neighbours
def dfs(binary_image, labels, i, j, curLab):
    stack = [(i, j)]
    while stack:
        i, j = stack.pop()

        if 0 <= i < binary_image.shape[0] and 0 <= j < binary_image.shape[1]:
            if binary_image[i][j] == 1 and labels[i][j] == 0:
                labels[i][j] = curLab

                for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if 0 <= i + x < binary_image.shape[0] and 0 <= j + y < binary_image.shape[1]:
                     stack.append((i + x, j + y))
    return labels


def classifyOring(labels, thresh):
    foreground_label = 1

    # calculate area and perimeter
    foreground_area, foreground_perimeter = calculate_area_and_perimeter(labels, foreground_label)

    print("Area of the foreground object:", foreground_area)
    print("Perimeter of the foreground object:", foreground_perimeter)

    circularity = (4 * np.pi * foreground_area) / (foreground_perimeter ** 2)
    print("Circularity:", circularity)

    circThresh = 0.19  # thresh, can be adjusted

    if circularity >= circThresh:
        return "Pass"
    else:
        return "Fail"


def calculate_area_and_perimeter(labels, foreground_label):
    foreground_area = np.sum(labels == foreground_label)
    perimeter = calculate_perimeter(labels, foreground_label)

    return foreground_area, perimeter


def calculate_perimeter(labels, foreground_label):
    # create a copy of the labeled image to store the perimeter mask
    perimeter_mask = np.zeros_like(labels, dtype=np.uint8)
    perimeter = 0

    # iterate over each pixel
    for i in range(1, labels.shape[0] - 1):
        for j in range(1, labels.shape[1] - 1):
            # check if the pixel is part of the foreground component and has at least one background neighbor
            if labels[i, j] == foreground_label and (labels[i-1, j] != foreground_label or
                                                    labels[i+1, j] != foreground_label or
                                                    labels[i, j-1] != foreground_label or
                                                    labels[i, j+1] != foreground_label):
                perimeter_mask[i, j] = 255
                perimeter += 1

    # show the perimeter mask
    cv.imshow('Perimeter Mask', perimeter_mask)

    return perimeter



def visualize_labeled_image(labels):
    # get unique labels only count forground labels
    unique_labels = np.unique(labels)[1:]

    color_map = {label: np.random.randint(0, 255, 3) for label in unique_labels}

    colored_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    for label in unique_labels:
        colored_image[labels == label] = color_map[label]

    # show the colored image
    cv.imshow('Colored Labeled Image', colored_image)

                
# read in an image into memory
for i in range(1,16):
    start_time = time.time()
    img = cv.imread('Orings/Oring' + str(i) +'.jpg',0)
    hist = imhist(img)

    # finding threshold, and add it to image
    thresh = find_peak(hist)-70
    print(thresh)
    
    thresh_img = threshold(img,thresh)
    
    # binary morphology called and displayed
    binImage = binaryMorphology(thresh_img)

    # connected component labeling
    labels = connectedComponentLabeling(binImage, thresh)

    # classify the oring
    classification = classifyOring(labels, thresh)

    # End measuring time
    end_time = time.time()
    processing_time = end_time - start_time
    processing_time_str = f"Processing Time: {processing_time:.2f} seconds"

    
    img = cv.cvtColor(thresh_img,cv.COLOR_GRAY2BGR)
    cv.imshow('original image',img)
    cv.putText(img,str(thresh),(10,30),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),thickness=2)
    cv.putText(img, processing_time_str, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

    cv.imshow('thresholded image',img)
    cv.imshow("bin Image", binImage)

    # check if connected component labeling returned null
    if labels is None:
        print("Error: Connected component labeling failed.")
    else:
        # cv.imshow('Labeled Image', labels.astype(np.uint8) * 100) # *100 controls visiblity for the labels, can be set to anything
        visualize_labeled_image(labels)
    

    print("Oring " + str(i) + " is a " + classification)

    cv.waitKey()
    plt.plot(hist)
    plt.show()
    
cv.destroyAllWindows()