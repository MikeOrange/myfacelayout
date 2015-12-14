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


class ImageElement(object):
    def __init__(self, filename):
        self.filename = filename
        self.img = None
        self.imgBGR = None
        self.width = 0
        self.height = 0
        self.load()
        self._gray = None
        self._mask = None
        self._mask_inv = None

    def load(self):
        self.img = cv2.imread(self.filename, -1)
        self.imgBGR = cv2.imread(self.filename, 1)
        if self.img == None:
            raise IOError(self.filename + " not found")
        self.height, self.width = self.imgBGR.shape[:2]

    @property
    def gray(self):
        if self._gray == None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def mask(self):
        if self._mask == None:
            ret, self._mask = cv2.threshold(
                self.gray,
                10, 255,
                cv2.THRESH_BINARY)
        return self._mask

    @property
    def mask_inv(self):
        if self._mask_inv == None:
            self._mask_inv = cv2.bitwise_not(self.mask)
        return self._mask_inv

    def get_resized(self, newWidth, newHeight):
        def resize_elem(elem):
            return cv2.resize(elem,
                              (newWidth, newHeight),
                              interpolation=cv2.INTER_AREA)
        imgBGR = resize_elem(self.imgBGR)
        mask = resize_elem(self.mask)
        mask_inv = resize_elem(self.mask_inv)
        return imgBGR, mask, mask_inv

    def show(self):
        cv2.imshow('img', self.imgBGR)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class BackgroundImage(object):
    def __init__(self, filename):
        self.filename = filename
        self.img = None
        self._gray = None
        self.width = 0
        self.height = 0
        self.load()
        self._faces = []

    def load(self):
        self.img = cv2.imread(self.filename,
                              cv2.CV_LOAD_IMAGE_COLOR)
        if self.img == None:
            raise IOError("Background file not found")
        self.height, self.width = self.img.shape[:2]

    @property
    def gray(self):
        if self._gray == None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    def show(self):
        cv2.imshow('img', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _detect_faces(self):
        def detect(gray, cascade):
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
            return rects
        if not self._faces:
            cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)
            self._faces = detect(self.gray, cascade)

    @property
    def faces(self):
        if not self._faces:
            self._detect_faces()
        return self._faces

    def _cut_excess(self):
        pass

    def put_image_over_background(self, image, directives):
        for (x, y, w, h) in self.faces:
            face = (x, y, w, h)
            x1, x2, y1, y2 = directives(image, face)
            elemWidth = x2-x1
            elemHeight = y2-y1

            import ipdb; ipdb.set_trace()

            img, mask, mask_inv = image.get_resized(elemWidth, elemHeight)

            # Cutting image if necessary
            if x1 < 0:
                img = img[:, -x1:]
                mask = mask[:, -x1:]
                mask_inv = mask_inv[:, -x1:]
                x1 = 0
            if y1 < 0:
                img = img[-y1:, :]
                mask = mask[-y1:, :]
                mask_inv = mask_inv[-y1:, :]
                y1 = 0

            if x2 > self.width:
                x2 = self.width
                img = img[:, :x2]
                mask = mask[:, :x2]
                mask_inv = mask_inv[:, :x2]

            if y2 > self.height:
                y2 = self.height
                img = img[:y2, :]
                mask = mask[:y2, :]
                mask_inv = mask_inv[:y2, :]

            roi = self.img[y1:y2, x1:x2]



            regionHeight = y2 - y1
            regionWidth = x2 - x1

            # roi_bg contains the original image only where the mustache is not
            # in the region that is the size of the mustache.
            roi_bg = np.zeros((regionHeight, regionWidth, 3), dtype=np.uint8)
            cv2.bitwise_and(roi, roi, dst=roi_bg, mask=mask_inv)

            roi_fg = np.zeros((regionHeight, regionWidth, 3), dtype=np.uint8)
            # roi_fg contains the image of the mustache only where the mustache is
            cv2.bitwise_and(img, img, dst=roi_fg, mask=mask)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            self.img[y1:y2, x1:x2] = dst




class PromotionalPicture(object):
    def __init__(self, background_file):
        self.background = BackgroundImage(background_file)
        self.main_outfit = None
        self.secondary_img = None
        self.message_img = None
        self._set_used_files()

    def _set_used_elements(self, main, secondary, message):
        try:
            self.main_outfit = ImageElement(main)
        except IOError as e:
            print e
        try:
            self.secondary_img = ImageElement(secondary)
        except IOError as e:
            print e
        try:
            self.message_img = ImageElement(message)
        except IOError as e:
            print e

    def _set_used_files(self):
        pass



class ChristmasPromo(PromotionalPicture):
    def __init__(self, background_file):
        super(ChristmasPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/christmas_hat.png"
        )
        secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT", ""
        )
        message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/navidad.png"
        )
        self._set_used_elements(main_outfit_file,
                                secondary_img_file,
                                message_img_file)

    def put_hat(self):
        def directive(used_image, face):
            x, y, w, h = face

            newWidth = int(3*w/4)
            newHeight = newWidth * used_image.height / used_image.width

            x1 = x + w/4
            x2 = x1 + newWidth
            y1 = y - newHeight/2
            y2 = y1 + newHeight

            return (x1, x2, y1, y2)

        self.background.put_image_over_background(self.main_outfit, directive)


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
        totalImgHeight, totalImgWidth = self.face_img.shape[:2]

        rects = self._detect_faces()

        for (x, y, w, h) in rects:
            hatWidth = int(3*w/4)
            hatHeight = hatWidth * origHatHeight / origHatWidth

            x1 = x + w/4
            x2 = x1 + hatWidth
            y1 = y - hatHeight/2
            y2 = y1 + hatHeight

            hatWidth = x2 - x1
            hatHeight = y2 - y1

            newHat = cv2.resize(hatBGR, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)

            # Cutting image if necessary
            if x1 < 0:
                newHat = newHat[:, -x1:]
                mask = mask[:, -x1:]
                mask_inv = mask_inv[:, -x1:]
                x1 = 0
            if y1 < 0:
                newHat = newHat[-y1:, :]
                mask = mask[-y1:, :]
                mask_inv = mask_inv[-y1:, :]
                y1 = 0

            if x2 > totalImgWidth:
                x2 = totalImgWidth
                newHat = newHat[:, :x2]
                mask = mask[:, :x2]
                mask_inv = mask_inv[:, :x2]

            if y2 > totalImgHeight:
                y2 = totalImgHeight
                newHat = newHat[:y2, :]
                mask = mask[:y2, :]
                mask_inv = mask_inv[:y2, :]

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
            #cv2.imwrite("result.png", hatBGR)

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