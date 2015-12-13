"""
File that includes class to make an image overlay over a background image
"""
import cv2
import cv2.cv as cv
import numpy as np
from os import environ

FACE_CASCADE_FILE = environ.get(
    "FACE_CASCADE_FILE",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

class PromoPicture(object):
    """
    Promotion picture including background image, profession overlay
    (hat, tool) and some text message on the front
    """
    def __init__(self, face_file):
        self.face_file = face_file
        self._set_used_files()
        self.face_img = self.read_image(self.face_file)
        self.main_outfit_img = None
        self.secondary_img = None
        self.message_img = None
        self.final_img = None

    def _set_used_files(self):
        self.main_outfit_file = None
        self.secondary_img_file = None
        self.message_img_file = None

    def show_original_image(self):
        cv2.imshow('img', self.face_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def read_image(filename, both=None):
        if not both:
            img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_COLOR)
            if img == None:
                raise IOError("Background file not found")
            return img
        else:
            img = cv2.imread(filename, -1)
            imgBGR = cv2.imread(filename, 1)
            if img == None:
                raise IOError("Background file not found")
            return (img, imgBGR)

    def _put_image_on_location(self, x, y):
        pass

    def _resize_image(self, height, width):
        pass

    def _detect_faces(self):
        """
        Returns a list of the detected faces in tuples
        containing (x, y, width and height) within the image
        :return:
        """
        def detect(gray, cascade):
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
            return rects

        cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)
        gray = cv2.cvtColor(self.face_img, cv.CV_BGR2GRAY)

        return detect(gray, cascade)

    def _detect_nose(self):
        pass

    def save_final_image(self, destination_file):
        pass


class ChristmasImage(PromoPicture):
    def __init__(self, face_file):
        super(ChristmasImage, self).__init__(face_file)

    def _set_used_files(self):
        self.main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/christmas_hat.png"
        )
        self.secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT"
        )
        self.message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/navidad.png"
        )

    def get_complete_image(self):
        hat, hatBGR = self.read_image(self.main_outfit_file, both=True)
        self.main_outfit_img = hat

        hat_gray = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)
        ret, orig_mask = cv2.threshold(hat_gray, 10, 255, cv2.THRESH_BINARY)
        orig_mask_inv = cv2.bitwise_not(orig_mask)

        origHatHeight, origHatWidth = hatBGR.shape[:2]

        rects = self._detect_faces()

        for (x, y, w, h) in rects:
            #cv2.rectangle(self.face_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            hatWidth = int(3*w/4)
            hatHeight = hatWidth * origHatHeight / origHatWidth

            x1 = x + w/4
            x2 = x1 + hatWidth
            y1 = y - hatHeight/2
            y2 = y1 + hatHeight

            print x1
            print x2
            print y1
            print y2

            hatWidth = x2 - x1
            hatHeight = y2 - y1

            newHat = cv2.resize(hatBGR, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)


            # Cutting image if necessary
            if x1 < 0:
                newHat = newHat[:, hatWidth+x1:]
                mask = mask[:, hatWidth+x1:]
                mask_inv = mask_inv[:, hatWidth+x1:]
                x1 = 0
            if y1 < 0:
                newHat = newHat[-y1:, :]
                mask = mask[-y1:, :]
                mask_inv = mask_inv[-y1:, :]
                y1 = 0
            """
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
            """

            # take ROI for mustache from background equal to size of mustache image
            roi = self.face_img[y1:y2, x1:x2]

            regionHeight = y2 - y1
            regionWidth = x2 - x1

            # roi_bg contains the original image only where the mustache is not
            # in the region that is the size of the mustache.
            roi_bg = np.zeros((regionHeight, regionWidth, 3), dtype=np.uint8)
            cv2.bitwise_and(roi, roi, dst=roi_bg, mask=mask_inv)

            roi_fg = np.zeros((regionHeight, regionWidth, 3), dtype=np.uint8)
            # roi_fg contains the image of the mustache only where the mustache is
            cv2.bitwise_and(newHat, newHat, dst=roi_fg, mask=mask)
            cv2.imwrite("result.png", hatBGR)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            self.face_img[y1:y2, x1:x2] = dst

        return self.final_img


class StylistImage(PromoPicture):
    def __init__(self, face_file):
        super(self.__class__).__init__(face_file)

    def _set_used_files(self):
        self.main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/christmas_hat.png"
        )
        self.secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT"
        )
        self.message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/navidad.png"
        )

    def get_complete_image(self):
        return self.final_img


class DesignerImage(PromoPicture):
    def __init__(self, face_file):
        super(self.__class__).__init__(face_file)

    def _set_used_files(self):
        self.main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/christmas_hat.png"
        )
        self.secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT"
        )
        self.message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/navidad.png"
        )


    def get_complete_image(self):
        return self.final_img


class ChefImage(PromoPicture):
    def __init__(self, face_file):
        super(self.__class__).__init__(face_file)

    def _set_used_files(self):
        self.main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/christmas_hat.png"
        )
        self.secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT"
        )
        self.message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/navidad.png"
        )

    def get_complete_image(self):
        return self.final_img