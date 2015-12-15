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
        if len(self._faces) == 0:
            self._detect_faces()
        return self._faces

    def _cut_excess(self):
        pass

    def put_image_over_background(self, image, directives):
        for (x, y, w, h) in self.faces:
            face = (x, y, w, h)
            x1, x2, y1, y2 = directives(image, face,
                                        self.width,
                                        self.height)
            elemWidth = x2-x1
            elemHeight = y2-y1

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
                img = img[:, :x2-x1]
                mask = mask[:, :x2-x1]
                mask_inv = mask_inv[:, :x2-x1]

            if y2 > self.height:
                y2 = self.height
                img = img[:y2-y1, :]
                mask = mask[:y2-y1, :]
                mask_inv = mask_inv[:y2-y1, :]

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

    def put_message(self):
        def directive(used_image, face, background_width, background_height):

            newWidth = int(background_width*7/8)
            newHeight = newWidth * used_image.height / used_image.width

            x1 = background_width - newWidth
            x2 = x1 + newWidth
            y1 = background_height - \
                 newHeight - int(background_height*0.05)
            y2 = y1 + newHeight

            return x1, x2, y1, y2

        self.background.put_image_over_background(self.message_img, directive)



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
        def directive(used_image, face, background_width, background_height):
            x, y, w, h = face

            newWidth = int(3*w/4)
            newHeight = newWidth * used_image.height / used_image.width

            x1 = x + w/4
            x2 = x1 + newWidth
            y1 = y - newHeight/2
            y2 = y1 + newHeight

            return (x1, x2, y1, y2)

        self.background.put_image_over_background(self.main_outfit, directive)


class CookPromo(PromotionalPicture):
    def __init__(self, background_file):
        super(CookPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/sombrero_chef.png"
        )
        secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT",
            "images/mustache2.png"
        )
        message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/disposicion.png"
        )
        self._set_used_elements(main_outfit_file,
                                secondary_img_file,
                                message_img_file)

    def put_hat(self):
        def directive(used_image, face, background_width, background_height):
            x, y, w, h = face

            newWidth = int(w)
            newHeight = newWidth * used_image.height / used_image.width

            x1 = x - w/7
            x2 = x1 + newWidth
            y1 = y - newHeight + y/8
            y2 = y1 + newHeight

            return (x1, x2, y1, y2)

        self.background.put_image_over_background(self.main_outfit, directive)


class FashionPromo(PromotionalPicture):
    def __init__(self, background_file):
        super(FashionPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/cintametrica2.png"
        )
        secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT",
            "images/tijeras.png"
        )
        message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/idea.png"
        )
        self._set_used_elements(main_outfit_file,
                                secondary_img_file,
                                message_img_file)

    def put_hat(self):
        def directive(used_image, face, background_width, background_height):
            x, y, w, h = face

            newWidth = w*2
            newHeight = newWidth * used_image.height / used_image.width

            x1 = x - newWidth/7
            x2 = x1 + newWidth
            y1 = y + h
            y2 = y1 + newHeight

            return (x1, x2, y1, y2)

        self.background.put_image_over_background(self.main_outfit, directive)

    def put_scissors(self):
        def directive(used_image, face, background_width, background_height):
            x, y, w, h = face

            newWidth = w*2/3
            newHeight = newWidth * used_image.height / used_image.width

            x1 = x + w
            x2 = x1 + newWidth
            y1 = y + h/2
            y2 = y1 + newHeight

            return (x1, x2, y1, y2)

        self.background.put_image_over_background(self.secondary_img, directive)


class HairdresserPromo(PromotionalPicture):
    def __init__(self, background_file):
        super(HairdresserPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = environ.get(
            "FACEOVERLAY_CHRISTMAS_HAT",
            "images/secador.png"
        )
        secondary_img_file = environ.get(
            "FACEOVERLAY_SNOW_EFFECT",
            "images/tijeras.png"
        )
        message_img_file = environ.get(
            "FACEOVERLAY_CHRISMAS_MESSAGE",
            "images/disposicion.png"
        )
        self._set_used_elements(main_outfit_file,
                                secondary_img_file,
                                message_img_file)

    def put_hat(self):
        def directive(used_image, face, background_width, background_height):
            x, y, w, h = face

            newWidth = w
            newHeight = newWidth * used_image.height / used_image.width

            x1 = x - newWidth/7
            x2 = x1 + newWidth
            y1 = y + h
            y2 = y1 + newHeight

            return (x1, x2, y1, y2)

        self.background.put_image_over_background(self.main_outfit, directive)

    def put_scissors(self):
        def directive(used_image, face, background_width, background_height):
            x, y, w, h = face

            newWidth = w*2/3
            newHeight = newWidth * used_image.height / used_image.width

            x1 = x + w
            x2 = x1 + newWidth
            y1 = y + h/2
            y2 = y1 + newHeight

            return (x1, x2, y1, y2)

        self.background.put_image_over_background(self.secondary_img, directive)