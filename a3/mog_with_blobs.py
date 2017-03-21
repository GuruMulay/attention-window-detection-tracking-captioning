#!/usr/bin/env python

import numpy as np
import cv2

blobDict = {}


class Blob:

    blobCount = 0

    def __init__(self, cnt_id, cnt, area, centroid_x, centroid_y, vel_x, vel_y):
        self.cnt_id = cnt_id
        self.cnt = cnt
        self.area = area
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.vel_x = vel_x
        self.vel_y = vel_y

        Blob.blobCount += 1

    def getBlobCount(self):
        return Blob.blobCount

    def getRectWindow(self):
        x, y, w, h = cv2.boundingRect(cnt)
        return x, y, w, h

    def printBlobAttributes(self):
        print "contourId: ", self.cnt_id, ", area: ", self.area, ", centroid: ", self.centroid_x, self.centroid_y, ", velocity: ", self.vel_x, self.vel_y


if __name__ == '__main__':
    print "main "

    # blobs = [1,9,3]
    #
    # for i, cid in enumerate(np.array(blobs)):
    #     print i, cid
    #     if cid in blobDict.keys():
    #         print "present"
    #         # blobDict[cid] += [kp_select[i]]
    #     else:
    #         blobDict[cid] = []
    #         print "initialized"
    #         blobDict[cid] += [Blob(1,100,7,5,5,1,1)]


    cap = cv2.VideoCapture('example1.mov')
    # fgbg = cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=20,detectShadows=True)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=150,nmixtures=5,backgroundRatio=.5,noiseSigma=1)

    frame_count = 0
    while(frame_count < 215):
        frame_count += 1

        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        kernel = np.ones((3,3),np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        # erosion = cv2.erode(fgmask,np.ones((3,3),np.uint8),iterations = 2)
        # blur = cv2.GaussianBlur(erosion,(5,5),0)

        # blur = cv2.GaussianBlur(opening, (11, 11), 0)
        blur = cv2.medianBlur(opening, 7)
        ret, thresh = cv2.threshold(blur, 10, 255, 0)

        filtered_im = thresh  #opening  #erosion

        im2, contours, hierarchy = cv2.findContours(filtered_im, 1, 2)
        print "detected contours", len(contours) # print contours


        # populate the dictionary per frame (***dictionary is cleared for every frame)
        for id, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            n = 0 # count of selected blobs out of detected blobs
            if area > 1000:
                M = cv2.moments(cnt)  # print M # moments of the contour
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])  # print "cenroid is", cx, cy

                blobDict[id] = []
                blobDict[id] += [Blob(id, cnt=cnt, area=area, centroid_x=cx, centroid_y=cy, vel_x=0, vel_y=0)]

                x, y, w, h = cv2.boundingRect(cnt)
                print "rectangle window:", x,y,w,h
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # cross verify the object
                print "blobDict[id]", blobDict[id][n].getRectWindow()

                n += 1


        print "frame number and its blob dictionary ", frame_count, blobDict
        # CLEAR the dictionary after every frame
        blobDict.clear()


        # cv2.imshow('fgmask',fgmask)
        # cv2.imshow('erosion',erosion)
        # cv2.imshow('opening',opening)
        # cv2.imshow('blur',blur)
        # cv2.imshow('rect frame',frame)

        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
