#!/usr/bin/env python

'''
MOSSE tracking sample

This sample implements correlation-based tracking approach, described in [1].

Usage:
  mosse.py [--pause] [<video source>] [iterations of erosion]

  --pause  -  Start with playback paused at the first video frame.
              Useful for tracking target selection.

  Draw rectangles around objects with a mouse to track them.

Keys:
  SPACE    - pause video
  c        - clear targets

[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~bolme/publications/Bolme2010Tracking.pdf
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2
import shapely.geometry as gs 
from common import draw_str, RectSelector
import video

import mog
import csv

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5



class MOSSE:
    MosseCounter = 0
    def __init__(self, frame, rect, index, prev_id=0, reInit=False):
        self.stationary_for_frames = 0
        x1, y1, x2, y2 = rect
        self.index = index
        if not reInit:            
            self.id_ = MOSSE.MosseCounter
            MOSSE.MosseCounter = MOSSE.MosseCounter + 1
        else:
            self.id_ = prev_id
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            self.stationary_for_frames = self.stationary_for_frames + 1
            return

        self.pos = x+dx, y+dy
        if abs(dx) + abs(dy) < 1:
            self.stationary_for_frames = self.stationary_for_frames + 1
        else:
            self.stationary_for_frames = 0
        
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        if self.stationary_for_frames < 40:
            (x, y), (w, h) = self.pos, self.size
            x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))        
            if self.good:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
            else:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
            draw_str(vis, (x1, y2+16), 'PSR: %.2f' % (self.psr))
            draw_str(vis, (x1, y1-6), 'Object: %d' % (self.id_))

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1

class MotionTracker:
    def __init__(self, video_src, paused = False):
        self.cap = video.create_capture(video_src)
        _, self.frame = self.cap.read()
        # self.out is the output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi',fourcc, 30,(self.frame.shape[1],self.frame.shape[0]))
        cv2.imshow('frame', self.frame)
        # For manually selecting objects to track. Not using, but not removing either.
        self.rect_sel = RectSelector('frame', self.onrect)
        self.trackers = []
        self.paused = paused

    def onrect(self, rect):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tracker = MOSSE(frame_gray, rect,len(self.trackers))
        self.trackers.append(tracker)

    def run(self):
        frame_number = 0
        csvf = open('tracks.csv', 'w')
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()                
                if not ret:
                    break
                # First update existing trackers with current frame
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                # Remove mosse trackers that are statinary for 100 frames
                for tracker in self.trackers:
                    if (tracker.stationary_for_frames > 100):
                        self.trackers.remove(tracker)
                # Calculate the union of all existing trackers
                # The type of object returned depends on the relationship between the operands. 
                # The union of polygons (for example) will be a polygon or a multi-polygon depending on whether they intersect or not.
                union_of_trackers = None
                index = 0
                for tracker in self.trackers:
                    # fix tracker index
                    tracker.index = index
                    index = index + 1
                    tracker.update(frame_gray)
                    (x, y), (w, h) = tracker.pos, tracker.size
                    # tracker returns center and size; convert it to topleft and bottomright points
                    x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
                    b = gs.box(x1,y1,x2,y2)
                    if not union_of_trackers:
                        union_of_trackers = b
                    else:
                        union_of_trackers = union_of_trackers.union(b)
                
                # Call MOG and get a list of large enough foregound rects
                rects = mog.getForegroundRects(self.frame)                
                for rect in rects:                    
                    x, y, w, h = rect                    
                    r = gs.box(x,y,x+w,y+h) # MOG returns topleft and size; need topleft and bottomright points
                    # check if this rect(MOG) is already mostly covered by other trackers(MOSSE)
                    if union_of_trackers:
                        common_area = union_of_trackers.intersection(r).area
                        ratio_covered = common_area/r.area
                        if ratio_covered > .6:
                            continue
                    
                    # check if this rect(MOG) almost contains another tracker(MOSSE), if yes then update that tracker's rect
                    new_rect = True
                    for tracker in self.trackers:
                        (x, y), (w, h) = tracker.pos, tracker.size
                        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
                        b = gs.box(x1,y1,x2,y2)
                        #if(r.area  > b.area):
                        common_area = r.intersection(b).area
                        ratio_covered = common_area/b.area
                        #print(ratio_covered)
                        if ratio_covered > .2:
                            if r.area / b.area > 1.2 or b.area / r.area > 1.2:
                                x1, y1, x2, y2 = r.union(b).bounds
                                self.trackers[tracker.index] = MOSSE(frame_gray, (int(x1), int(y1), int(x2), int(y2)), tracker.index, tracker.id_, reInit=True)
                            else:
                                tracker.stationary_for_frames = 0
                            new_rect = False
                            break
                            
                    # otherwise new tracker found; append to the list of trackers
                    if new_rect:
                        x, y, w, h = rect
                        tracker = MOSSE(frame_gray, (x,y,x+w,y+h), len(self.trackers))
                        self.trackers.append(tracker)

                # write csv output file
                for tracker in self.trackers:
                    (x, y), (w, h) = tracker.pos, tracker.size
                    x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
                    if x1 < 0: 
                        x1 = 0
                    if y1 < 0:
                        y1 = 0
                    if x2 > self.frame.shape[1]-1:
                        x2 = self.frame.shape[1]-1
                    if y2 > self.frame.shape[0]-1:
                        y2 = self.frame.shape[0]-1
                    writer = csv.writer(csvf)
                    writer.writerow((tracker.index, frame_number, x1, y1, x2, y2))

                # print("self.trackers", self.trackers)
                # print("frame number", frame_number)
                frame_number += 1

            vis = self.frame.copy()
            for tracker in self.trackers:
                tracker.draw_state(vis)
            #if len(self.trackers) > 0:
            #    cv2.imshow('tracker state', self.trackers[-1].state_vis)
            self.rect_sel.draw(vis)

            cv2.imshow('frame', vis)
            self.out.write(vis)
            ch = cv2.waitKey(10) % 256
            if ch == 27:
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []


if __name__ == '__main__':
    print (__doc__)
    #a = gs.box(0,0,10,10)
    #b = gs.box(10,10,20,20)
    #c = gs.box(5,5,15,15)
    #print(a.union(b).intersection(c).area)
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['pause'])
    opts = dict(opts)
    try:
        video_src = args[0]
    except:
        video_src = '0'

    app = MotionTracker(video_src, paused = '--pause' in opts)
    app.run()    
    app.out.release()
