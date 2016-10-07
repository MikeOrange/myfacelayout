"""
File that includes class to make an image overlay over a background image
Heavily influenced by http://sublimerobots.com/2015/02/dancing-mustaches/
and http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
"""
import cv2
import cv2.cv as cv
import numpy as np
from os import environ


FACE_CASCADE_FILE = environ.get(
    "FACE_CASCADE_FILE",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")


def draw_rects(img, rects, color):
    """
    Draws a rectangle in the specified area of the image
    :param img: Image to be modified
    :param rects: list of tuples containing coordinates
    :param color: color of the rectangles
    """
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


class ImageElement(object):
    """
    Represents an image to be mounted over a picture, like a hat
    or glasses
    """
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
        """
        Loads image from the filesystem
        """
        self.img = cv2.imread(self.filename, -1)
        self.imgBGR = cv2.imread(self.filename, 1)
        if self.img is None:
            raise IOError(self.filename + " not found")
        self.height, self.width = self.imgBGR.shape[:2]

    @property
    def gray(self):
        """
        Returns a copy of the loaded image in grayscale for manipulation
        """
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def mask(self):
        """
        Separates the region of the image which interests us
        """
        if self._mask is None:
            ret, self._mask = cv2.threshold(
                self.gray,
                10, 255,
                cv2.THRESH_BINARY)
        return self._mask

    @property
    def mask_inv(self):
        """
        Returns the inverted region of the mask (it's background)
        """
        if self._mask_inv is None:
            self._mask_inv = cv2.bitwise_not(self.mask)
        return self._mask_inv

    def get_resized(self, newWidth, newHeight):
        """
        Returns resized image
        :param newWidth: width of the resized image
        :param newHeight: height of the resized image
        :return: Tuple containing the image, its mask and inverted mask
        """
        def resize_elem(elem):
            return cv2.resize(elem,
                              (newWidth, newHeight),
                              interpolation=cv2.INTER_AREA)
        imgBGR = resize_elem(self.imgBGR)
        mask = resize_elem(self.mask)
        mask_inv = resize_elem(self.mask_inv)
        return imgBGR, mask, mask_inv

    def show(self):
        """
        Shows the image on screen, for debugging purposes
        """
        cv2.imshow('img', self.imgBGR)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class BackgroundImage(object):
    """
    Represents the image where the mask will be drawn, in this
    case pictures of people containing faces
    """
    def __init__(self, filename):
        self.filename = filename
        self.img = None
        self._gray = None
        self.width = 0
        self.height = 0
        self.load()
        self._faces = []

    def load(self):
        """
        Loads the image from the filesystem
        """
        self.img = cv2.imread(self.filename,
                              cv2.CV_LOAD_IMAGE_COLOR)
        if self.img is None:
            raise IOError("Background file not found")
        self.height, self.width = self.img.shape[:2]

    @property
    def gray(self):
        """
        Returns a copy of the loaded image in grayscale to allow for
        tasks like face detection
        """
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    def show(self):
        """
        Shows the loaded image, used for debugging purposes
        """
        cv2.imshow('img', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _detect_faces(self):
        """
        Detects the regions where faces are found in the image by using
        opencv's face cascade file
        """
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
        """
        Returns a list of tuples containing the coordinates of the
        faces withing the picture
        """
        if len(self._faces) == 0:
            self._detect_faces()
        return self._faces

    def put_image_over_background(self, image, directives):
        """
        Draws an image over our loaded BackgroundImage
        :param image: ImageElement instance of the image to be put over
        the background
        :param directives: callback function used to determine where the image
        is going to be drawn relative to the faces detected
        """
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

            # roi_bg contains the original image without the bits where
            # the mounted image will be
            roi_bg = np.zeros((regionHeight, regionWidth, 3), dtype=np.uint8)
            cv2.bitwise_and(roi, roi, dst=roi_bg, mask=mask_inv)

            # roi_fg contains the mounted image only on that region
            roi_fg = np.zeros((regionHeight, regionWidth, 3), dtype=np.uint8)
            cv2.bitwise_and(img, img, dst=roi_fg, mask=mask)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            self.img[y1:y2, x1:x2] = dst

    def save_image(self):
        """
        Saves modified image to the filesystem
        """
        cv2.imwrite(self.filename, self.img)


class PromotionalPicture(object):
    """
    Base class for all the masking combinations available for users
    contains images and directives for every element used
    """
    def __init__(self, background_file):
        self.background = BackgroundImage(background_file)
        self.main_outfit = None
        self.secondary_img = None
        self.message_img = None
        self._set_used_files()

    def _set_used_elements(self, main, secondary, message):
        """
        Instantiates the ImageElements to be mounted on the background
        """
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
        """
        Common image for all promos, footer message containing reference
        to the promotional webpage
        """
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
    """
    Mounts a christmas hat over the detected faces
    """
    def __init__(self, background_file):
        super(ChristmasPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = "res/christmas_hat.png"
        secondary_img_file = ""
        message_img_file = "res/label.png"
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
    """
    Mounts a cook hat over the detected faces
    """
    def __init__(self, background_file):
        super(CookPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = "res/sombrero_chef.png"
        secondary_img_file = "res/mustache2.png"
        message_img_file = "res/label.png"
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
    """
    Puts a measuring tape around detected people's necks and can also
    put scissors on the side
    """
    def __init__(self, background_file):
        super(FashionPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = "res/cintametrica2.png"
        secondary_img_file = "res/tijeras.png"
        message_img_file = "res/label.png"
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
    """
    Able to put a hairdryer and scissors close to the person detected in
    the picture
    """
    def __init__(self, background_file):
        super(HairdresserPromo, self).__init__(background_file)

    def _set_used_files(self):
        main_outfit_file = "res/secador.png"
        secondary_img_file = "res/tijeras.png"
        message_img_file = "res/label.png"
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