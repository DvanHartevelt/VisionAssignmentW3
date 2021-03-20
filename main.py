# Vision week 3 assignments

import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import ProgressBar as pb
import time

def main():


    """
    Importing a leopard print. I would like to count the spots here.
    """

    imgLeopard = cv2.imread("Resources/leopard.jpeg", 0)
    imgLeopard = cv2.resize(imgLeopard, (int(imgLeopard.shape[1]/4), int(imgLeopard.shape[0]/4)))

    # plt.hist(imgLeopard.ravel(), 256, [0,256])
    # plt.show()
    # threshold = int(input("What should be the threshold?")) #36

    (thres, imgLeopardBW) = cv2.threshold(imgLeopard, 36, 255, cv2.THRESH_BINARY_INV)
    imgLeopardINV = cv2.bitwise_not(imgLeopardBW)

    cv2.imwrite("Output/LeopardThreshold.png", imgLeopardBW)
    cv2.imwrite("Output/LeopardThresholdINV.png", imgLeopardINV)

    #and now, to import a man
    imgMan = cv2.imread("Resources/man.jpeg", 0)
    (thres, imgManBW) = cv2.threshold(imgMan, 230, 255, cv2.THRESH_BINARY_INV)

    def erosion(img, kernel, blackForeground=False):
        if blackForeground:
            return dilation(img, kernel)
        else:
            imgNew = np.zeros_like(img)

            for y in range(1, img.shape[0] - 1):
                for x in range(1, img.shape[1] - 1):
                    newValue = np.bitwise_or(img[y - 1: y + 2, x - 1: x + 2], np.logical_not(kernel))
                    imgNew[y, x] = np.all(newValue) * 255

            return imgNew

    """
    #Dialation
    """
    def dilation(img, kernel, blackForeground=False):
        if blackForeground:
            return erosion(img, kernel)
        else:
            imgNew = np.zeros_like(img)

            for y in range(1, img.shape[0] - 1):
                for x in range(1, img.shape[1] - 1):
                    newValue = np.bitwise_and(img[y - 1: y + 2, x - 1: x + 2], kernel)
                    imgNew[y, x] = np.any(newValue) * 255

            return imgNew

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    starttime = time.time()
    imgDilate = dilation(imgLeopardBW, kernel)
    homebrewDilateTime = (time.time() - starttime)

    imgDilateINV = dilation(imgLeopardINV, kernel, blackForeground=True)

    starttime = time.time()
    imgCV2dil = cv2.dilate(imgLeopardBW, kernel, iterations=1)
    cv2DilateTime = (time.time() - starttime)

    print(f"My own dilation function took {homebrewDilateTime} s, cv2's took {cv2DilateTime} s.")

    cv2.imwrite("Output/LeopardDialate.png", imgDilate)
    cv2.imwrite("Output/LeopardDialateINV.png", imgDilateINV)

    cv2.imshow("Original", imgLeopard)
    cv2.imshow("Black and White", imgLeopardBW)
    cv2.imshow("Black and White, black foreground", imgLeopardINV)
    cv2.imshow("Dilation, my function", imgDilate)
    cv2.imshow("Dilation, black foreground", imgDilateINV)
    cv2.imshow("Dilation, cv2", imgCV2dil)

    cv2.waitKey(0)


    """
    #Erotion
    """
    def erotion(img, kernel, blackForeground=False):
        if blackForeground:
            return dilation(img, kernel)
        else:
            imgNew = np.zeros_like(img)

            for y in range(1, img.shape[0] - 1):
                for x in range(1, img.shape[1] - 1):
                    newValue = np.bitwise_or(img[y - 1: y + 2, x - 1: x + 2], np.logical_not(kernel))
                    imgNew[y, x] = np.all(newValue) * 255

            return imgNew

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    starttime = time.time()
    imgErote = erotion(imgLeopardBW, kernel)
    homebrewErotionTime = (time.time() - starttime)

    imgEroteINV = erotion(imgLeopardINV, kernel, blackForeground=True)

    starttime = time.time()
    imgCV2ero = cv2.dilate(imgLeopardBW, kernel, iterations=1)
    cv2ErotionTime = (time.time() - starttime)

    print(f"My own erosion function took {homebrewErotionTime} s, cv2's took {cv2ErotionTime} s.")

    cv2.imwrite("Output/LeopardErote.png", imgErote)
    cv2.imwrite("Output/LeopardEroteINV.png", imgEroteINV)

    cv2.imshow("Original", imgLeopard)
    cv2.imshow("Black and White", imgLeopardBW)
    cv2.imshow("Black and White, black foreground", imgLeopardINV)
    cv2.imshow("Erotion, my function", imgErote)
    cv2.imshow("Erotion, black foreground", imgEroteINV)
    cv2.imshow("Erotion, cv2", imgCV2ero)

    cv2.waitKey(0)
    

    """
    #Opening and closing
    """

    #like before, open and close are eachothers inverse
    def open(img, kernel, iterations = 1, blackForeground = False):
        if blackForeground:
            return close(img, kernel, iterations=iterations)
        else:
            interim = cv2.erode(img, kernel, iterations=iterations)
            return cv2.dilate(interim, kernel, iterations=iterations)

    def close(img, kernel, iterations=1, blackForeground=False):
        if blackForeground:
            return open(img, kernel, iterations=iterations)
        else:
            interim = cv2.dilate(img, kernel, iterations=iterations)
            return cv2.erode(interim, kernel, iterations=iterations)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    imgOpened = open(imgLeopardBW, kernel)
    imgClosed = close(imgLeopardBW, kernel)

    interim = close(imgOpened, kernel, iterations=2)
    final = open(interim, kernel, iterations=2)

    cv2.imwrite("Output/LeopardOpened.png", imgOpened)
    cv2.imwrite("Output/LeopardClosed.png", imgClosed)
    cv2.imwrite("Output/LeopardFinal.png", final)

    cv2.imshow("Original", imgLeopardBW)
    cv2.imshow("Opened", imgOpened)
    cv2.imshow("Closed", imgClosed)
    cv2.imshow("final", final)

    cv2.waitKey(0)


    """
    #Making a skeleton
    """
    def getSkeleton(img):
        size = np.size(img)
        skel = np.zeros_like(img, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        done = False
        timesTried = 0

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True

            timesTried += 1

            if timesTried > 1000:
                break

        return skel

    temp = cv2.morphologyEx(imgManBW, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8),iterations=2)
    skeleton = getSkeleton(temp)

    cv2.imshow("Original", imgManBW)
    cv2.imshow("Closed", temp)
    cv2.imshow("Skeleton", skeleton)

    cv2.waitKey(0)

    print("end of the program.")


if __name__ == '__main__':
    main()
