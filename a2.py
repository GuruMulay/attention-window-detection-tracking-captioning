#!/usr/bin/env python

import cv2
import sys
import numpy as np
from operator import attrgetter
import window_history


def keypoint_to_window(kp):
    """Converts the position of keypoint to a square window using its 'size'

    Args:
        kp: The keypoint as used in OpenCV SIFT

    Returns:
        Window in the form of a 4-tuple for x,y coordinates for top-left and bottom right corners
    """
    
    x, y = kp.pt
    r = kp.size/2

    x0, y0 = (x-r, y-r)
    x1, y1 = (x+r, y+r)

    return (x0, y0, x1, y1)



def extract_window_from_frame(w, frame):
    """Uses an affine transform to extract the attention window from frame

    Args:
        w: A 4-tupe representing a window using the top-left and bottom right (x,y) coordinates
        frame: The frame from which to extract the attention window

    Returns:
        Part of the frame demarcated by window 'w'
    """
    
    src_points = np.array( [ (aw[0], aw[1]), (aw[0], aw[3]), (aw[2], aw[3]) ] , np.float32)
    dest_points = np.array( [ (0, 0), (0, aw[3] - aw[1]), (aw[2] - aw[0], aw[3] - aw[1]) ] , np.float32)

    t = cv2.getAffineTransform(src_points, dest_points)

    out_frame = cv2.warpAffine(frame, t, (int(aw[2] - aw[0]), int(aw[3] - aw[1])))

    return out_frame



def get_keypoint_attrs(k):
    """Extracts individual SIFT keypoint details from its octave attribute

    Args:
        k: A keypoint (singular) yielded by SIFT

    Returns:
        (octave, layer, scale): where octave, layer, and scale have usual meanings
    """
    
    octave = k.octave & 255
    layer = (k.octave >> 8) & 255
    octave = octave if octave < 128 else (-128 | octave) # http://code.opencv.org/issues/2987
    scale = 1 / float(1 << octave) if octave >= 0 else (float)(1 << -octave)

    return (octave, layer, scale)



input_video = cv2.VideoCapture(sys.argv[1])

# Exit if can't open the input file
if not input_video.isOpened():
    print "Can't open the input video {}".format(sys.argv[1])
    sys.exit(0)

sift = cv2.xfeatures2d.SIFT_create()

cv2.namedWindow("Sift", cv2.WINDOW_AUTOSIZE)

frame_counter = 0

while True:
    read_success, frame = input_video.read()
        
    if read_success:
        print 'Advancing to next frame'
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #frame_gray = cv2.Canny(frame_gray, 100, 200, 7)
        
        kps = sift.detect(frame_gray, None)

        kps.sort(key=attrgetter('size', 'response'), reverse=True)

        frame_with_kp = frame

        frame_attention_window = None
        
        for i in range(len(kps)):

            aw = keypoint_to_window(kps[i])
            
            octave, layer, scale= get_keypoint_attrs(kps[i])

            if window_history.add_if_new(keypoint_to_window(kps[i]), scale):
                frame_with_kp = None
                frame_with_kp = cv2.drawKeypoints(frame, [ kps[i] ], frame_with_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                print octave, layer, scale

                frame_attention_window = aw
                frame_counter += 1
                break

        if frame_attention_window != None:            
            cv2.imwrite('results/' + str(frame_counter) + '.jpg', extract_window_from_frame(frame_attention_window, frame))

        cv2.imshow("Sift", frame_with_kp)

        # Use 'q' to stop the processing
        # and any other key to progress to next frame
        if cv2.waitKey(0) == ord('q'):
            break
    else:
        print 'Reached end of video. Stopping...'
        break

cv2.destroyWindow("Sift")
input_video.release()
