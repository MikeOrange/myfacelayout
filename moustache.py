import cv2
import cv2.cv as cv
import numpy as np

def detect(gray, cascade):
    rects = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def put_hat_on_beckham(source_image):
    img = cv2.imread(source_image, cv2.CV_LOAD_IMAGE_COLOR)  # Read image file

    imgMustache = cv2.imread("mustache.png", -1)
    imgMustacheBGR = cv2.imread("mustache.png", 1)

    # Create mask for the moustache
    #orig_mask = imgMustache[:, :, 3]

    # Create inverted mask for the mustache
    #orig_mask_inv = cv2.bitwise_not(orig_mask)

    # NEW CODE
    mustache_gray = cv2.cvtColor(imgMustache, cv2.COLOR_BGR2GRAY)
    ret, orig_mask = cv2.threshold(mustache_gray, 10, 255, cv2.THRESH_BINARY)
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    ##############################################

    #imgMustache = imgMustacheBGR
    origMustacheHeight, origMustacheWidth = imgMustacheBGR.shape[:2]

    cascade = cv2.CascadeClassifier(
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)

    rects = detect(gray, cascade)

    for (x, y, w, h) in rects:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect a nose within the region bounded by each face (the ROI)
        noseCascade = cv2.CascadeClassifier(
            "/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml")
        nose = noseCascade.detectMultiScale(roi_gray)

        for (nx, ny, nw, nh) in nose:
            # The mustache should be three times the width of the nose
            mustacheWidth =  3 * nw
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

            # Center the mustache on the bottom of the nose
            x1 = nx - (mustacheWidth/4)
            x2 = nx + nw + (mustacheWidth/4)
            y1 = ny + nh - (mustacheHeight/2)
            y2 = ny + nh + (mustacheHeight/2)

            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            # Re-calculate the width and height of the mustache image
            mustacheWidth = x2 - x1
            mustacheHeight = y2 - y1

            print "W", mustacheWidth
            print "H", mustacheHeight

            #import ipdb; ipdb.set_trace()

            # Re-size the original image and the masks to the mustache sizes
            # calculated above
            mustache = cv2.resize(imgMustacheBGR, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)

            # take ROI for mustache from background equal to size of mustache image
            roi = roi_color[y1:y2, x1:x2]

            # roi_bg contains the original image only where the mustache is not
            # in the region that is the size of the mustache.
            roi_bg = np.zeros((mustacheHeight, mustacheWidth, 3), dtype=np.uint8)
            cv2.bitwise_and(roi, roi, dst=roi_bg, mask=mask_inv)

            roi_fg = np.zeros((mustacheHeight, mustacheWidth, 3), dtype=np.uint8)
            # roi_fg contains the image of the mustache only where the mustache is
            cv2.bitwise_and(mustache, mustache, dst=roi_fg, mask=mask)
            cv2.imwrite("result.jpg", roi_fg)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('img', img)
        #cv2.imwrite("result.jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
     #put_hat_on_beckham("larger_image.jpg")
     put_hat_on_beckham("maburro.png")