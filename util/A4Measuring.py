import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

scale = 3
lP = 297
wP = 210

def measureObjects(img):
    def getContours(img, Thres=[100,100], showCanny=False, minArea=1000, filter=0, draw=False, Histogram=False):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.GaussianBlur(imgGray, (5,5), 3)
        cv2.imshow("gray", imgGray)
        if Histogram: plt.hist(imgGray.ravel(),256,[0,256]); plt.show()
        imgCanny = cv2.Canny(img, Thres[0], Thres[1])

        kernel = np.ones((2,2))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=3)
        imgThres = cv2.dilate(imgDil,kernel, iterations=2)

        contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if showCanny:
            cv2.imshow("Canny", imgCanny)
            cv2.imshow("Thres", imgThres)
            cv2.waitKey(0)

        finalContours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > minArea:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour,0.02*peri, True)
                bbox = cv2.boundingRect(approx)
                if filter > 0:
                    if len(approx) == filter:
                        finalContours.append([len(approx), area, approx, bbox, contour])
                else:
                    finalContours.append([len(approx), area, approx, bbox, contour])

        finalContours = sorted(finalContours, key= lambda x:x[0], reverse=True)

        if draw:
            for con in finalContours:
                cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

        return img, finalContours

    def reorder(mypoints):
        newMypoints = np.zeros_like(mypoints)
        mypoints = mypoints.reshape((4, 2))

        add = np.sum(mypoints, 1)
        newMypoints[0] = mypoints[np.argmin(add)]
        newMypoints[3] = mypoints[np.argmax(add)]

        diff = np.diff(mypoints, 1)
        newMypoints[1] = mypoints[np.argmin(diff)]
        newMypoints[2] = mypoints[np.argmax(diff)]

        return newMypoints

    def warpImg(img, points, w, h, pad=20):
        reorderedPoints = reorder(points)
        pts1 = np.float32(reorderedPoints)
        pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])

        warpMatrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, warpMatrix, (w,h))
        imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]

        return imgWarp

    img, contours = getContours(img, Thres=[195,200],minArea=50000, filter=4)
    if len(contours) != 0:
        biggest = contours[0][2]
        imgWarped = warpImg(img, biggest, wP*scale, lP*scale)
        imgContours, contours2 = getContours(imgWarped, minArea=1000, Thres=[50,90], filter=4)

        if len(contours2) != 0:
            for contour in contours2:
                # cv2.polylines(imgContours, [contour[2]], True, (0, 255, 0), 3)
                points = reorder(contour[2])
                width = math.dist(points[0][0], points[1][0]) // scale
                length = math.dist(points[0][0], points[2][0]) // scale
                # print(f"object width is {width}, length is {length}.")
                blue = (255, 0, 0)
                x, y, w, h = contour[3]
                cv2.arrowedLine(imgWarped, tuple(points[0][0]), tuple(points[1][0]), blue, 3, 8, 0, 0.05)
                cv2.arrowedLine(imgWarped, tuple(points[0][0]), tuple(points[2][0]), blue, 3, 8, 0, 0.05)
                cv2.putText(imgWarped, '{} mm'.format(width), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, blue, 1)
                cv2.putText(imgWarped, '{} mm'.format(length), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, blue, 1)


    return imgContours, [[12, 2], [14, 5]]



def main():
    imgOriginal = cv2.imread("../Resources/A4.jpeg")
    imgResized = cv2.resize(imgOriginal, (0,0) , None, 0.5, 0.5)

    img, objectsMeasurements = measureObjects(imgResized)

    cv2.imshow("Original", imgResized)
    cv2.imshow("Measure overlay", img)

    for i in range(len(objectsMeasurements)):
        print(f"Object nr {i} has length {objectsMeasurements[i][0]} and width {objectsMeasurements[i][1]}.")

    cv2.waitKey(0)


    pass

if __name__ == "__main__":
    main()