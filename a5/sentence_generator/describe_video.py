import cv2
import csv
import numpy as np
import sys
# from a2 import window_history

video_file = 'example1.mov'
width = 1280
height = 720
csv_file = 'classes2.csv'

moving = {}
still = {}

def _get_common_window(window_new, window_old):
    """ Returns the overlap of two windows if possible

    Args:
        window_new: the new window
        window_old: the old window

    Returns:
        A 4-tuple represnting the window formed from the overlap of input windows
    """
    xn0, yn0, xn1, yn1 = window_new
    xo0, yo0, xo1, yo1 = window_old

    # No overlap
    if (xn1 < xo0) or (xn0 > xo1) or (yn1 < yn0) or (yn0 > yo1):
        return None
    else:
    # Overlap
        return ( max(xn0, xo0), max(yn0, yo0),  min(xn1, xo1), min(yn1, yo1))


def _get_overlap(window_new, window_old):
    """Get new window's overlap with the old window

    Args:
        window_new: a 4-tuple representing new window
        window_old: a 4-tuple representing old window

    Returns:
        A floating point value between 0.0 (no overlap) and 1.0 (new window inside old window)

    """

    def area(rect):
        x0, y0, x1, y1 = rect
        return (x1 - x0) * (y1 - y0)

    common_window = _get_common_window(window_new, window_old)

    if common_window == None:
        return 0.0
    else:
        return area(common_window) / float(area(window_old) + area(window_new) - area(common_window))


def populate_objects_dictionary(csv_file):
    """Creates a dictionary for still and moving objects respectively

        Args:
            csv_file: the output csv from VGG (assignment 4)

        Returns:
            nothing

    """
    # print "describing video..."
    n_moving = 1
    n_still = 1
    with open(csv_file, 'rb') as csvfile:
        csvfile = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(csvfile)
        for i, row in enumerate(csvfile):
            object = row[7].split(',')[0]  # removes synonyms
            row = [row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), object, float(row[8])]
            # print i, row

            if row[0]=='moving':
                moving[n_moving] = []
                moving[n_moving] += row
                # print moving[n_moving]
                dx = row[5]-row[3]
                dy = row[6]-row[4]
                frames_diff = row[2]-row[1]

                if np.abs(dx) > np.abs(dy) and dx > 0:
                    direction = "towards right"
                elif np.abs(dx) > np.abs(dy) and dx < 0:
                    direction = "towards left"
                elif np.abs(dy) > np.abs(dx) and dy > 0:
                    direction = "towards lower edge"
                elif np.abs(dy) > np.abs(dx) and dy < 0:
                    direction = "towards upper edge"

                motion = "slowly" if ((np.maximum(np.abs(dx), np.abs(dy))/frames_diff) < 4) else "relatively fast"
                row_new = [dx, dy, frames_diff, direction, motion]  # add dx, dy, frames_diff to dict
                moving[n_moving] += row_new
                # print moving[n_moving]

                n_moving+=1


            if row[0] == 'still':
                # print row[7]
                # for key in still:
                #     print "key so far", key, _get_overlap((still[key][3], still[key][4], still[key][5], still[key][6]), (row[3], row[4], row[5], row[6]))
                #     if _get_overlap((still[key][3], still[key][4], still[key][5], still[key][6]), (row[3], row[4], row[5], row[6])) > 0.5:
                #         print "overlapping"
                still[n_still] = []
                still[n_still] += row
                n_still += 1


def describe_video_scene():
    """Writes sentenceStill descriptions for both still and moving objects (background and foreground)
         Args:
             none
         Returns:
             none
     """
    # print moving, still

    # Describe still, background in the video
    begining = "In the background, there is "  # since we are describing the still objects
    sentenceStill = ""

    for key in still:
        if still[key][8]>0.25:  # activation > threshold

            object = still[key][7]
            # print object
            cx = (still[key][5]+still[key][3])/2
            cy = (still[key][6]+still[key][4])/2
            # print key, still[key][5], still[key][3], still[key][6], still[key][4], cx, cy

            if (0 < cx < width/3) and (0 < cy < height/3):
                position = "the upper left, "
            elif (width/3 < cx < 2*width/3) and (0 < cy < height/3):
                position = "the upper middle, "
            elif (2*width/3 < cx < width) and (0 < cy < height/3):
                position = "the upper right, "
            elif (0 < cx < width/3) and (height/3 < cy < 2*height/3):
                position = "the middle left, "
            elif (width/3 < cx < 2*width/3) and (height/3 < cy < 2*height/3):
                position = "the center, "
            elif (2*width/3 < cx < width) and (height/3 < cy < 2*height/3):
                position = "the middle right, "
            elif (0 < cx < width/3) and (2*height/3 < cy < height):
                position = "the lower left, "
            elif (width/3 < cx < 2*width/3) and (2*height/3 < cy < height):
                position = "the lower middle, "
            elif (2*width/3 < cx < width) and (2*height/3 < cy < height):
                position = "the middle right, "

            sentenceStill += "a " + object + " in " + position
    sentenceStill = begining + sentenceStill[:-2] + "."  # add fullstop, remove trailing comma
    print sentenceStill


    # Describe moving, foreground in the video
    sentenceMotion = "In the foreground, "
    moving_right = []
    moving_left = []
    moving_lower_edge = []
    moving_upper_edge = []

    for key in moving:
        object = moving[key][7]
        # print object
        if moving[key][12] in ["towards right"]:
            # print moving[key][12]
            moving_right.append(key)
        elif moving[key][12] in ["towards left"]:
            # print moving[key][12]
            moving_left.append(key)
        elif moving[key][12] in ["towards upper edge"]:
            # print moving[key][12]
            moving_upper_edge.append(key)
        elif moving[key][12] in ["towards lower edge"]:
            # print moving[key][12]
            moving_lower_edge.append(key)

    # print moving_right, moving_left, moving_upper_edge, moving_lower_edge

    if moving_right and moving_left:  # if not empty
        rightMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_right]
        # print rightMotion
        leftMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_left]
        # print leftMotion
        sentenceMotion += ', '.join(rightMotion) + "AND " + ', '.join(leftMotion) + "are moving in opposite direction."

    elif moving_right or moving_left:  # if not empty
        rightMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_right]
        # print rightMotion
        leftMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_left]
        # print leftMotion
        sentenceMotion += ', '.join(leftMotion) + "" + ', '.join(rightMotion) + "are moving in same direction."

    else:
        a = 1
        # print ""

    if moving_upper_edge and moving_upper_edge:  # if not empty
        upMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_upper_edge]
        # print rightMotion
        downMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_lower_edge]
        # print leftMotion
        sentenceMotion += ', '.join(upMotion) + "AND " + ', '.join(downMotion) + "are moving in opposite direction."

    elif moving_upper_edge or moving_upper_edge:  # if not empty
        upMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_upper_edge]
        # print rightMotion
        downMotion = ["a " + moving[k][7] + " (moving " + moving[k][12] + " " + moving[k][13] + ") " for k in moving_lower_edge]
        # print leftMotion
        sentenceMotion += ', '.join(upMotion) + "" + ', '.join(downMotion) + "are moving in same direction."

    else:
        a = 1
        # print ""

    print sentenceMotion



if __name__ == '__main__':
    # input_video = cv2.VideoCapture(video_file)
    # print "(width, height) of the frame: ", input_video.get(cv2.CAP_PROP_FRAME_WIDTH), input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    csv_file = sys.argv[1]
    populate_objects_dictionary(csv_file=csv_file)
    describe_video_scene()