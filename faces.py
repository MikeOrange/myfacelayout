import cv2
import cv2.cv as cv
import sys
import numpy

# img is actually gray
def detect(img, cascade):
    rects = cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=3,
        minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def put_hat_on_beckham(my_file):
    img = cv2.imread(my_file, cv2.CV_LOAD_IMAGE_COLOR)  # Read image file
    if (img == None):
        print "Could not open or find the image"
    else:
        #imgHat = cv2.imread("christmas_hat.png", -1)# cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #imgHat2 = cv2.imread("christmas_hat.png", 1)
        imgHat = cv2.imread("mustache.png", -1)# cv2.CV_LOAD_IMAGE_GRAYSCALE)
        imgHat2 = cv2.imread("mustache.png", 1)

        # Crate mask for the hat (?)
        orig_mask = imgHat[:, :, 3]
        orig_mask_inv = cv2.bitwise_not(orig_mask)

        imgHat = imgHat2
        origHatHeight, origHatWidth = imgHat.shape[:2]

        cascade = cv2.CascadeClassifier(
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = detect(gray, cascade)

        for (x, y, w, h) in rects:
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]

            # The hat should be the same width that the face
            hatWidth = w
            hatHeight = hatWidth*origHatHeight/origHatWidth

            roi_gray = gray[y:y+hatHeight, x:x+hatWidth]
            roi_color = img[y:y+hatHeight, x:x+hatWidth]

            x1 = x
            x2 = x + hatWidth
            y1 = y
            y2 = y + hatHeight
            print (x1, x2, y1, y2)
            # Check for clipping

            """
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
            """

            hatWidth = x2 - x1
            hatHeight = y2 - y1

            #import ipdb; ipdb.set_trace()

            #Put the hat over the face
            newHat = cv2.resize(imgHat, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)

            roi = roi_color#[y1:y2, x1:x2]

            # roi_bg contains the original image only where the mustache is not
            # in the region that is the size of the mustache.
            roi_bg = cv2.bitwise_and(img[y:y+hatHeight, x:x+hatWidth], img[y:y+hatHeight, x:x+hatWidth], mask=mask_inv)

            # roi_fg contains the image of the mustache only where the mustache is
            roi_fg = cv2.bitwise_and(newHat, newHat, mask=mask)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            #roi_color[y1:y2, x1:x2] = dst
            img[y:y+hatHeight, x:x+hatWidth] = dst

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        # Extract face coordinates
        x1 = rects[0][3]
        y1 = rects[0][0]
        x2 = rects[0][4]
        y2 = rects[0][5]
        y=y2-y1
        x=x2-x1
        # Extract face ROI
        faceROI = gray[x1:x2, y1:y2]

        # Show face ROI
        cv2.imshow('Display face ROI', faceROI)
        small = cv2.imread("average_face.png",cv2.CV_LOAD_IMAGE_COLOR)
        print "here"
        small=cv2.resize(small, (x, y))
        cv2.namedWindow('Display image')          # create window for display
        cv2.imshow('Display image', small)        # Show image in the window

        print "size of image: ", img.shape        ## print size of image
        cv2.waitKey(1000)
        """

def put_mustache_on_me(my_file):
    # location of OpenCV Haar Cascade Classifiers:
    baseCascadePath = "/usr/share/opencv/haarcascades/"

    # xml files describing our haar cascade classifiers
    faceCascadeFilePath = baseCascadePath + "haarcascade_frontalface_default.xml"
    noseCascadeFilePath = baseCascadePath + "haarcascade_mcs_nose.xml"

    # build our cv2 Cascade Classifiers
    faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
    noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)

    #-----------------------------------------------------------------------------
    #       Load and configure mustache (.png with alpha transparency)
    #-----------------------------------------------------------------------------

    # Load our overlay image: mustache.png
    imgMustache = cv2.imread('mustache.png', -1)

    # Create the mask for the mustache
    orig_mask = imgMustache[:, :, 3]

    # Create the inverted mask for the mustache
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    # Convert mustache image to BGR
    # and save the original image size (used later when re-sizing the image)
    imgMustache = imgMustache[:, :, 0:3]
    origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

    #-----------------------------------------------------------------------------
    #       Main program loop
    #-----------------------------------------------------------------------------

    # collect video input from first webcam on system
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture video feed
        ret, frame = video_capture.read()

        # Create greyscale image from the video feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in input video stream
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

       # Iterate over each face found
        for (x, y, w, h) in faces:
            # Un-comment the next line for debug (draw box around all faces)
            # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect a nose within the region bounded by each face (the ROI)
            nose = noseCascade.detectMultiScale(roi_gray)

            for (nx, ny, nw, nh) in nose:
                # Un-comment the next line for debug (draw box around the nose)
                #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)

                # The mustache should be three times the width of the nose
                mustacheWidth = 3 * nw
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

                # Re-size the original image and the masks to the mustache sizes
                # calcualted above
                mustache = cv2.resize(imgMustache,
                                      (mustacheWidth, mustacheHeight),
                                      interpolation=cv2.INTER_AREA)
                mask = cv2.resize(orig_mask,
                                  (mustacheWidth, mustacheHeight),
                                  interpolation=cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv,
                                      (mustacheWidth, mustacheHeight),
                                      interpolation=cv2.INTER_AREA)
                """
                # take ROI for mustache from background equal to size of mustache image
                roi = roi_color[y1:y2, x1:x2]

                # roi_bg contains the original image only where the mustache is not
                # in the region that is the size of the mustache.
                roi_bg = cv2.bitwise_and(roi, roi ,mask=mask_inv)

                # roi_fg contains the image of the mustache only where the mustache is
                roi_fg = cv2.bitwise_and(mustache, mustache, mask=mask)

                # join the roi_bg and roi_fg
                dst = cv2.add(roi_bg, roi_fg)

                # place the joined image, saved to dst back over the original image
                roi_color[y1:y2, x1:x2] = dst
                """
                break

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # press any key to exit
        # NOTE;  x86 systems may need to remove: "& 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    if len(sys.argv) != 2:  # Check for error in usage syntax
        print "Usage : python faces.py <image_file>"
    else:
        put_hat_on_beckham(sys.argv[1])
        #put_mustache_on_me("")