#!/usr/bin/env python
import cv2
import sys
import math
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) == 3:
        # Set up the input video file
        input_video = cv2.VideoCapture(sys.argv[1])

        # Exit if can't open the input file
        if not input_video.isOpened():
            print "Can't open the input video {}".format(sys.argv[1])
            sys.exit(0)
       
        # Ask the user of begin and end frames
        start_frame = input("Enter the first frame (0 based) where the object becomes visible: ")
        end_frame = input("Enter the last frame (0 based) where the object is visible: ")
        
        # Ask the user of the bounding box in the first frame
        # x is the column number, y is the row number, both relative to the upper left corner
        x, y, width, height = input("Enter bounding box description (x, y, width, height) [comma separated]: ")

        if x < 0 or y < 0 or width <= 0 or height <= 0:
            print "Invalid description of bounding box"
            sys.exit(0)

        # Ask the user of the bounding box velocity
        # Note that velocity can be floating point value
        # with right and down pointed components
        vel_x, vel_y = input("Enter the velocity (columns_per_frame (x/f), rows_per_frame (y/f)) [comma separated]: ")

        out_width, out_height = input("Enter the (width, height) of ouput video [comma separated]: ")

        # If we can get the fps of the input video, then fine, otherwise assume 30 fps for output
        fps = input_video.get(cv2.cv.CV_CAP_PROP_FPS) if input_video.get(cv2.cv.CV_CAP_PROP_FPS) > 0.0 else 30

        # Assume MJPG output
        # Apparently the Gstreamer backend for OpenCV is broken
        output_video = cv2.VideoWriter(sys.argv[2], cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, (out_width, out_height))        

        # These will change due to velocity
        cur_x = x
        cur_y = y

        # To keep track of temporal bounds
        frame_index = 0
        while True:
            read_success, frame = input_video.read()

            if read_success:
                if frame_index > end_frame:
                    print "Reached end of the temporal range. Stopping"
                    break
                elif frame_index >= start_frame:
                    # If the current frame is within the input temporal range

                    # Get input video height and width
                    # Only place where height precedes width
                    # Normally, opencv functions work with width first, height second
                    input_height, input_width = frame.shape[:2]

                    # Rounding off to pixel positions
                    cur_x_rounded = round(cur_x)
                    cur_y_rounded = round(cur_y)

                    if cur_x_rounded >= 0 and cur_y_rounded >= 0 and cur_x_rounded + width - 1 < input_width and cur_y_rounded + height - 1 < input_height:
                        src_points = np.array([(cur_x_rounded, cur_y_rounded), (cur_x_rounded + width - 1, cur_y_rounded + height - 1), (cur_x_rounded, cur_y_rounded + height - 1)], np.float32)
                        dest_points = np.array([(0, 0), (out_width - 1, out_height - 1), (0, out_height - 1)], np.float32)
                      
                        transform_mat = cv2.getAffineTransform(src_points, dest_points)
                        warped_frame = cv2.warpAffine(frame, transform_mat, (out_width, out_height))
                        output_video.write(warped_frame)
                    else:
                        print "Bounding box {} exceeds input frame size {}. Stopping...".format((cur_x_rounded, cur_y_rounded, cur_x_rounded + width - 1, cur_y_rounded + height - 1), (input_width, input_height))
                        break
                        
                    cur_x += vel_x
                    cur_y += vel_y
            else:
                print "Reached end of the video. Stopping"
                break
            
            # Go to next frame
            frame_index += 1
                                       
                
        input_video.release()
        output_video.release()
        
    else:
        print "Usage: python a1.py <input_video> <output_video>"
