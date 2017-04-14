#!/usr/bin/env python

import cv2
import sys
import numpy as np
from operator import attrgetter, itemgetter
import window_history
from clustering1 import cluster_keypoints
import cluster_dbscan


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


def extract_window_from_frame(aw, frame):
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


def process_naive(frame, sift):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kps = sift.detect(frame_gray, None)
    kps.sort(key=attrgetter('size', 'response'), reverse=True)

    best_keypoints = []
    frame_attention_window = None

    for i in range(len(kps)):
        aw = keypoint_to_window(kps[i])
        octave, layer, scale = get_keypoint_attrs(kps[i])

        if window_history.add_if_new(aw, scale):
            frame_attention_window = aw
            best_keypoints += [kps[i]]
            break

    assert frame_attention_window != None
    return (frame_attention_window, best_keypoints)


def process_naive2(frame, sift):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kps = sift.detect(frame_gray, None)

    sz_and_rsp = np.array([[k.size for k in kps], [k.response for k in kps]], dtype=np.float32)

    sz_and_rsp[0] /= np.max(sz_and_rsp[0])
    sz_and_rsp[1] /= np.max(sz_and_rsp[1])

    scores = sz_and_rsp[0] * sz_and_rsp[1]

    # Sort keypoints based on score
    sorted_kps = map(None, kps, scores)

    sorted_kps.sort(key=itemgetter(1), reverse=True)

    best_keypoints = []
    frame_attention_window = None

    for kp, score in sorted_kps:
        aw = keypoint_to_window(kp)
        octave, layer, scale = get_keypoint_attrs(kp)

        if window_history.add_if_new(aw, scale):
            frame_attention_window = aw
            best_keypoints += [ kp ]
            break

    assert frame_attention_window != None
    return (frame_attention_window, best_keypoints)


# Register your implementation here
# Key is the name of the implementation
# Value is the function that actually implements it
# Don't forget to add your implementation to the valid names
# mentioned in documentation of process() for consistency

_impls = {
    'naive': process_naive,
    'naive2': process_naive2,
    'cluster1': cluster_keypoints,
    'dbscan': cluster_dbscan.cluster
}


def process(impl, frame, *args):
    """Process the frame to yield an attention window and corresponding one or more keypoints

    Args:
        impl: The name of methodology to be executed. Valid names are [ 'naive', 'naive2', 'cluster1', 'dbscan']
        frame: The frame yielded by cv2.VideoCapture
        args: Extra arguments needed by the function that actually implements the underlying methodolody

    Returns:
        A 2-tuple constituting the attention window and list of corresponding keypoints
    """
    if impl in _impls.keys():
        return _impls[impl](frame, *args)
    else:
        raise ValueError('Unimplemented implementation name passed as argument')

class FeatureDetector:
    def __init__(self, process='dbscan'):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.process = process
    
    def get_window(self,frame):
        aw, kps = process(self.process, frame, self.sift)
        return extract_window_from_frame(aw, frame), aw





# The main thread follows below
if __name__ == '__main__':
    input_video = cv2.VideoCapture(sys.argv[1])

    # Exit if can't open the input file
    if not input_video.isOpened():
        print "Can't open the input video {}".format(sys.argv[1])
        sys.exit(0)

    sift = cv2.xfeatures2d.SIFT_create()
    #bms = BMS(opening_width=13, dilation_width=1, normalize=False)
    #bms = BMS()

    cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)

    frame_counter = 0
    while True:
        read_success, frame = input_video.read()
        if len(sys.argv)>2 and frame_counter < int(sys.argv[2]):
            frame_counter += 1
            continue

        if read_success:
            print 'Advancing to next frame'

            # Replace 'naive' with your own implementation
            # which should accept a frame returned by cv2.VideoCapture
            # and optionally one or more extra arguments if needed
            # Be sure to link your implementation in _impls
            # aw, kps = process('naive', frame, sift)
            aw, kps = process('dbscan', frame, sift)

            frame_with_kps = frame.copy()
            num_colors = len(cluster_dbscan.colors)
            for i in list(range(0, 50)):
                class_i = [kp for kp in kps if kp.class_id == i]
                if len(class_i) > 0:
                    frame_with_kps = cv2.drawKeypoints(frame_with_kps, class_i, frame_with_kps, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = cluster_dbscan.colors[i%num_colors])

            class_i = [kp for kp in kps if kp.class_id == -1]
            frame_with_kps = cv2.drawKeypoints(frame_with_kps, class_i, frame_with_kps, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = [255,255,255])
            frame_with_kps = cv2.rectangle(frame_with_kps, (int(aw[0]), int(aw[1])), (int(aw[2]), int(aw[3])) , (0,255,255),3)

            frame_counter += 1

            cv2.imshow("Test", frame_with_kps)
            cv2.imwrite('results/' + str(frame_counter) + '.jpg', extract_window_from_frame(aw, frame))

            # Use 'q' to stop the processing
            # and any other key to progress to next frame
            if cv2.waitKey(1000) == ord('q'):
                break
        else:
            print 'Reached end of video. Stopping...'
            break

    cv2.destroyWindow("Test")
    input_video.release()
