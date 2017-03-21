import numpy as np
import cv2

video_src = 'example1.mov'
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=500,nmixtures=5,backgroundRatio=.5,noiseSigma=1)
#fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=16,detectShadows=True)

fgmask=None
opening=None
erosion=None
blur = None
thresh = None

def getForegroundRects(frame):
    frame = frame.copy()
    global fgmask, opening, erosion, blur, thresh
    fgmask = fgbg.apply(frame)
    kernel = np.ones((5,5),np.uint8)
    #erosion = cv2.erode(fgmask,kernel,iterations = 1)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    erosion = cv2.erode(opening,np.ones((7,1),np.uint8),iterations = 10)
    
    blur = cv2.GaussianBlur(erosion,(99,99),0)
    
    ret, thresh = cv2.threshold(blur,30,255,0)
    im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            rects.append(cv2.boundingRect(cnt))
    
    return rects


if __name__ == '__main__':    
    cap = cv2.VideoCapture(video_src)
    
    while(1):
        ret, frame = cap.read()
        if ret:            
            rects = getMotionRect(frame)
            for r in rects:
                x,y,w,h = r
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.imshow('fgmask',fgmask)    
            cv2.imshow('erosion',erosion)
            cv2.imshow('blur',blur)
            cv2.imshow('opening',opening)
            cv2.imshow('threshold',thresh)
            cv2.imshow('frame',frame)
            

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        else:
            break
    
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
