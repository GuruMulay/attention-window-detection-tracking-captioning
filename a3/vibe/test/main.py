import cv2
import numpy as np
import vibe
import sys


def get_foreground_rects(foreground):
    kernel = np.ones((5,5),np.uint8)

    opening = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    #erosion = cv2.erode(opening, np.ones((7,1), np.uint8), iterations=10)
    #blur = cv2.GaussianBlur(erosion,(99,99),0)
    thresh = cv2.threshold(opening, 30, 255, 0)[1]

    cv2.imshow("processed", thresh)
    contours = cv2.findContours(thresh, 1, 2)[1]

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            rects.append(cv2.boundingRect(cnt))

    return rects


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(0)

    filename = sys.argv[1]

    cap = cv2.VideoCapture(filename)
    v = vibe.VIBE(3, 20, 4, 17, 2, 16)

    ret, frame = cap.read()
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
    mframe = vibe.Mat.from_array(frame)

    v.init(mframe)

    gray = mframe.clone()

    while True:
        v.update(gray)

        foreground = np.asarray(v.getMask())
        rects = getForegroundRects(foreground)

        for r in rects:
            x1, y1, w, h = r
            x2, y2 =  x1+w, y1+h
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)

        cv2.imshow("frame", frame)
        cv2.imshow("foreground", foreground)

        ret, frame = cap.read()
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
        gray = vibe.Mat.from_array(frame)

        key = cv2.waitKey(10)
        if key == 27:
            break
