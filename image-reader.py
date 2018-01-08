import cv2
from imutils import contours
import imutils
import sys
import math
from functools import *
import numpy as np
import cv2.text as text
from PIL import Image
import pytesseract


class BillReader:
    def __init__(self, image_path):
        self.debug = False
        self.image = cv2.imread(image_path, -1)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        (self.width, self.height) = self.image.shape[:2]

        # initialize a rectangular (wider than it is tall) and square
        # structuring kernel
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        self.sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def to_text_string(self):
        grad_x = self.gray.copy()
        grad_x = cv2.Canny(grad_x, 100, 200)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, self.sqKernel)
        lines = cv2.HoughLinesP(grad_x, 1, np.pi / 2, 300, None,
                                100, 10)

        lines_image = grad_x.copy()

        del grad_x

        lines_image[:] = (255)

        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(lines_image, (l[0], l[1]),
                     (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)

        im2, cnts, hierarchy = cv2.findContours(lines_image, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
        del lines_image

        cnts = [cnt for cnt in cnts if self.__is_table_cell(
            cnt, self.width, self.height)]
        cnts = self.__soft_contours(cnts)

        self.__debug_image(self.image, cnts)

        for idx, cnt in enumerate(cnts):
            if idx >= 10:
                continue
            (x, y, w, h) = cv2.boundingRect(cnt)
            section = self.__prepare_readable_section(self.gray, (x, y, w, h))
            pil_img = Image.fromarray(section)
            txt = pytesseract.image_to_string(
                pil_img, config="--psm 11 --oem 1")
            print("------------%s----------------" % idx)
            print(txt)
            self.imshow(section)

    def __soft_contours(self, cnts):
        def compare_cnt(cnt1, cnt2):
            (x1, y1, w1, h1) = cv2.boundingRect(cnt1)
            (x2, y2, w2, h2) = cv2.boundingRect(cnt2)
            if math.fabs(y1 - y2 ) < 10:
                return x1 - x2
            else:
                return y1 - y2

        cnts = sorted(cnts, key=cmp_to_key(compare_cnt))

        return cnts

    def __debug_image(self, image, cnts):
        if self.debug:
            for idx, cnt in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(image, (x, y),
                              (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, '%s' % idx, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            self.imshow(image)

    def __draw_contours(self, image, cnts):
        self.debug = True
        self.__debug_image(image, cnts)
        self.debug = False

    def __prepare_readable_section(self, gray, rect):
        '''chop_minimum_text_image'''
        (x, y, w, h) = rect
        section = gray[y:y + h, x:x + w]

        im_show = section.copy()

        im_show = cv2.morphologyEx(im_show, cv2.MORPH_TOPHAT, self.rectKernel)
        im_show = cv2.Sobel(im_show, cv2.CV_32F, dx=1, dy=0, ksize=-1)
        im_show = np.absolute(im_show)
        (minVal, maxVal) = (np.min(im_show), np.max(im_show))
        im_show = (255 * ((im_show - minVal) / (maxVal - minVal)))
        im_show = im_show.astype("uint8")
        im_show = cv2.morphologyEx(im_show, cv2.MORPH_CLOSE, self.rectKernel)
        im_show = cv2.threshold(im_show, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        im_show = cv2.morphologyEx(im_show, cv2.MORPH_CLOSE, self.sqKernel)
        im2, cnts, hierarchy = cv2.findContours(im_show, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)

        def bottom(cnt):
            (x, y, w, h) = cv2.boundingRect(cnt)
            return y + h + 10

        last_bottom = np.max([bottom(cnt) for cnt in cnts])

        section = section[0:last_bottom, 0:w]
        section = cv2.resize(section, None, fx=3, fy=3)
        return section

    ''' Check rect size to decide if the rect is a table cell '''

    def __is_table_cell(self, cnt, width, height):
        (x, y, w, h) = cv2.boundingRect(cnt)
        return w > 20 and h > 20 and (w * h < width * height / 2)

    def imshow(self, img):
        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    reader = BillReader(sys.argv[1])
    reader.to_text_string()
