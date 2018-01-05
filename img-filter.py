import cv2
from imutils import contours
import imutils
import sys
import numpy as np
import cv2.text as text
from PIL import Image
import pytesseract


def main(argv):
    image_path = argv[1]

    image = cv2.imread(image_path, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (width, height) = image.shape[:2]

    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    gradX = gray.copy()

    # 1. blackhat
    gradX = cv2.Canny(gradX, 100, 200)

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, sqKernel)

    lines = cv2.HoughLinesP(gradX, 1, np.pi / 2, 300, None,
                            100, 10)

    lines_image = gradX.copy()

    del gradX

    lines_image[:] = (255)

    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(lines_image, (l[0], l[1]),
                 (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)

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

    im2, cnts, hierarchy = cv2.findContours(lines_image, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)
    del lines_image

    def is_acceptable_rect(cnt):
        (x, y, w, h) = cv2.boundingRect(cnt)
        return w > 20 and h > 20 and (w * h < width * height / 2)

    cnts = [cnt for cnt in cnts if is_acceptable_rect(cnt)]
    cnts.reverse()

    for idx, cnt in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, '%s' % idx, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    chars_image = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, sqKernel)
    imshow(chars_image)

    chars_image = chars_image
 
    for idx, cnt in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(cnt)
        section = chars_image[y:y+h, x:x+w]
        im_show = section.copy()        
        section = cv2.resize(section,None,fx=3, fy=3)
        pil_img = Image.fromarray(section)
        txt = pytesseract.image_to_string(pil_img, config = "--oem 1")
        print("------------%s----------------" % idx)
        print(txt)
        imshow(im_show)

def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main(sys.argv)
