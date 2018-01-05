import cv2
from imutils import contours
import imutils
import sys
import numpy as np
from matplotlib import pyplot as plt

def main(argv):
    image_path = argv[1]

    image = cv2.imread(image_path, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    '''
    # 1. blackhat
    gradX = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    cv2.namedWindow('image',  cv2.WINDOW_NORMAL)
    imshow(gradX)
    '''
    gradX = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
    imshow(gradX)



    # 3. morph_close
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    imshow(gradX)
    

    # 4. threshold
    gradX = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    imshow(gradX)

    '''
    # 5. morph_close
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, sqKernel)
    imshow(gradX)
    '''
    '''
    chars_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 20)
    '''

    '''
    RETR_EXTERNAL 	
    retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.

    RETR_LIST 	
    retrieves all of the contours without establishing any hierarchical relationships.

    RETR_CCOMP 	
    retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.

    RETR_TREE 	
    retrieves all of the contours and reconstructs a full hierarchy of nested contours.

    RETR_FLOODFILL 	
    '''
    im2, cnts, hierarchy = cv2.findContours(gradX, cv2.RETR_LIST,
	                            cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        print("%s, %s, %s, %s" % (y, h, x, w))
    imshow(image)

def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv)