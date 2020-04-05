import cv2
import numpy as np
from base import img

def preprocessImage(img):
    #converting the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #applying Gaussain Blurring to reduce noise
    dst = cv2.GaussianBlur(gray,(1,1),cv2.BORDER_DEFAULT)

    #applying Inverse Binary Threshold
    ret,thresh_inv = cv2.threshold(gray, 180, 255,cv2.THRESH_BINARY_INV)

    return thresh_inv

def probHoughTransformUtil(thresh_inv):
    #reading in the raw image
    global img
    
    #applying probabilistic hough transform on the binary image
    minLineLength = 100
    maxLineGap = 60
    lines = cv2.HoughLinesP(thresh_inv,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for l in lines:
        x1,y1,x2,y2 = l[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2, cv2.LINE_AA)
    cv2.imwrite('hough.jpg',img)

    #applying contour to find the maximum area of the image (the 9x9 grid)
    img_hough = cv2.imread('hough.jpg',0) #reading in as grayscale
    contours,hierarchy = cv2.findContours(img_hough,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # finding the biggest area
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    poly_approx = cv2.approxPolyDP(cnt, epsilon, True)
    return poly_approx

#function takes in points
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[0]
    rect[2] = pts[2]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[3] = pts[3]
    rect[1] = pts[1]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1],
        [maxWidth - 1, 0]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


thresh_inv = preprocessImage(img)
poly_approx = probHoughTransformUtil(thresh_inv)
#perspective transformed image
img_PT = four_point_transform(thresh_inv,poly_approx)