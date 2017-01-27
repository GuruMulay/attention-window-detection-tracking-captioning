#!/usr/bin/env python
import cv2
import sys
import math
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) == 3:
        # Set up input and output video files
        input_video = cv2.VideoCapture(sys.argv[1])
       
        # May want to ensure that the files exist later on

        # Ask the user of begin and end frames
        start_frame = input('Enter the first frame (0 based) where the object becomes visible: ')
        input_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start_frame)
        
        end_frame = input('Enter the last frame (0 based) where the object is visible: ')
                                       

        # Ask the user of the bounding box in the first frame
        x, y, height, width = input('Enter bounding box description (x, y, height, width) [comma separated]: ')

        vel_x, vel_y = input('Enter the velocity (rows_per_second, columns_per_second) [comma separated]: ')

        out_height, out_width = input('Enter the height and width of ouput video: ')

        # The output video is going to have same codec, fps as the input video, but the resolution is changed as asked by user
        output_video = cv2.VideoWriter(sys.argv[2], int(input_video.get(cv2.cv.CV_CAP_PROP_FOURCC)), int(input_video.get(cv2.cv.CV_CAP_PROP_FPS)), (out_width, out_height))

        # These will change due to velocity
        cur_x = x
        cur_y = y

        while True:
            frame_captured, frame = input_video.read()
            if frame_captured:
                frame_index = int(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                if frame_index <= end_frame:
                    # If the current frame is within the input temporal range

                    # Get input video height and width
                    input_height, input_width = frame.shape[:2]

                    # Rounding off to pixel positions
                    cur_x_rounded = round(cur_x)
                    cur_y_rounded = round(cur_y)

                    if cur_x_rounded >= 0 and cur_y_rounded >= 0 and cur_x_rounded + width < input_width and cur_y_rounded + height < input_height:
                        src_points = np.array([(cur_x_rounded, cur_y_rounded), (cur_x_rounded + width, cur_y_rounded + height), (cur_x_rounded, cur_y_rounded + height)], np.float32)

                        dest_points = np.array([(0, 0), (out_width, out_height), (0, out_height)], np.float32)
                      
                        transform_mat = cv2.getAffineTransform(src_points, dest_points)
                        warped_frame = cv2.warpAffine(frame, transform_mat, (out_width, out_height))
                        output_video.write(warped_frame)
                    else:
                        print 'Invalid bounding box {} over input frame size {}. Stopping...'.format((cur_x_rounded, cur_y_rounded, cur_x_rounded + width, cur_y_rounded + height), (input_width, input_height))
                        break
                        
                    cur_x += vel_x
                    cur_y += vel_y
                else:
                    # We reached the end of temporal extent
                    print "Done"
                    break
            else:
                print "Reached the end of the input video"
                # We reached the end of the video itself
                break
                                       
                
        input_video.release()
        output_video.release()
        
    else:
        print 'Usage: python a1.py <input_video> <output_video>'
