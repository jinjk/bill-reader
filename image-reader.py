import cv2
from imutils import contours
import imutils
import sys
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
        cnts.reverse()

        self.__debug_image(cnts)

        chars_image = cv2.morphologyEx(
            self.gray, cv2.MORPH_BLACKHAT, self.sqKernel)
        self.imshow(chars_image)

        chars_image = chars_image

        for idx, cnt in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(cnt)
            section = self.__prepare_readable_section(self.gray, (x, y, w, h))
            pil_img = Image.fromarray(section)
            txt = pytesseract.image_to_string(pil_img, config="--oem 1")
            print("------------%s----------------" % idx)
            print(txt)
            self.imshow(section)

    def __debug_image(self, cnts):
        if self.debug:
            for idx, cnt in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(self.image, (x, y),
                              (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(self.image, '%s' % idx, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            self.imshow(self.image)

    def __prepare_readable_section(self, gray, rect):
        '''chop_minimum_text_image'''
        (x, y, w, h) = rect
        section = gray[y:y + h, x:x + w]
        im_show = section.copy()
        im_show = cv2.morphologyEx(im_show, cv2.WRITE_HAT, self.sqKernel)
        self.imshow(im_show)
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
