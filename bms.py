#!/usr/bin/env python
import random
import cv2
import numpy as np

class BMS:

    def __init__(self, frame, dw1, ow, nm, hb):
        self.frame = frame
        self.dilation_width = dw1
        self.opening_width = ow
        # Boolean
        self.normalize = nm
        # Boolean
        self.handle_border = hb

        frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        self.feature_maps = cv2.split(frame_lab)

        self.saliency_map = np.zeros(frame.shape[:2], np.float64)

        
    def generate_boolean_maps(self, step):
        for f in self.feature_maps:
            # To assign limits for thresholding
            min_val, max_val = cv2.minMaxLoc(f)[:2]

            for thresh in range(int(min_val), int(max_val), step):
                # Assign 1 to a pixel if greater than threshold otherwise 0
                bm = cv2.threshold(f, thresh, 1, cv2.THRESH_BINARY)[1]
                bm_opened = self.open_boolean_map(bm)
                self.saliency_map +=  self.generate_attention_map(bm_opened)
                # Generate inverse boolean map
                bm = cv2.bitwise_not(bm)
                bm_opened = self.open_boolean_map(bm)
                self.saliency_map += self.generate_attention_map(bm_opened)

                
    def open_boolean_map(self, bm):
        """Applies an opening operation on the raw boolean map

        Args:
            bm: A boolean map
        
        Returns:
            If opening width > 0, then a new boolean map with opening operation applied, else the same boolean map as input (not a copy)
        """
        bm_opened = None
        if self.opening_width > 0:
            bm_opened = cv2.morphologyEx(bm, cv2.MORPH_OPEN, None, bm_opened, (1,-1), self.opening_width)
        if bm_opened != None:
            return bm_opened
        else:
            return bm

    
    def generate_attention_map(self, bm):
        am = bm.copy()
        h, w = am.shape[:2]

        # Mask out all the pixels connected to borders
        if self.handle_border:
            for row in range(h):
                # All zero means no boundaries for flood filling
                # mask will change after every floodFill by setting the
                # filled pixels to a new value as set in the flags
                # which is 1 by default
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                jump = random.randint(5, 25) if random.random()>0.99 else 0
                if am[row, 0+jump] != 1:
                    cv2.floodFill(am, mask, (0+jump, row), (1,), (0,), (0,), 8) 
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                jump = random.randint(5, 25) if random.random()>0.99 else 0
                if am[row, w-1-jump] != 1:
                    cv2.floodFill(am, mask, (w-1-jump, row), (1,), (0,), (0,), 8)

            for col in range(w):
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                jump = random.randint(5, 25) if random.random()>0.99 else 0
                if am[0+jump, col] != 1:
                    cv2.floodFill(am, mask, (col, 0+jump), (1,), (0,), (0,), 8) 
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                jump = random.randint(5, 25) if random.random()>0.99 else 0
                if am[h-1-jump, col] != 1:
                    cv2.floodFill(am, mask, (col, h-1-jump), (1,), (0,), (0,), 8)
                    
        else:
            for row in range(h):
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
		if am[row, 0] != 1 :
                    cv2.floodFill(am, mask, (0, row), (1,), (0,), (0,), 8)
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                if am[row, w-1] != 1:
		    cv2.floodFill(am, mask, (w-1, row), (1,), (0,), (0,), 8)
	    for col in range(w):
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
		if am[0, col] != 1:
		    cv2.floodFill(am, mask, (col,0), (1,), (0,), (0,), 8)
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
		if am[w-1, col] != 1:
		    cv2.floodFill(am, mask, (col,h-1), (1,), (0,), (0,), 8)
            
        #min_val, max_val = cv2.minMaxLoc(am)[0:2] 
        
        am = (am != 1).astype(np.uint8) * 255

        if self.dilation_width > 0:
            am = cv2.dilate(am, None, am, (-1, -1), self.dilation_width)

        am = am.astype(np.float64)

        if self.normalize:
            cv2.normalize(am, am, 1.0, 0.0, cv2.NORM_L2)
        else:
            cv2.normalize(am, am, 1.0, 0.0, cv2.NORM_MINMAX)

        return am

    def get_saliency_map(self):
        out = np.zeros(self.saliency_map.shape[:2], dtype=np.float64)
        cv2.normalize(self.saliency_map, out, 255.0, 0.0, cv2.NORM_MINMAX)
        out = out.astype(np.uint8)
        return out
