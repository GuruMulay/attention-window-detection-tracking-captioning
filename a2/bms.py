#!/usr/bin/env python

import random
import cv2
import numpy as np


class BMS:

    def __init__(self, sample_step_size=8, opening_width=5, dilation_width=7, normalize=True, handle_border=True):

        self.sample_step_size = sample_step_size
        self.opening_width = opening_width
        self.dilation_width = dilation_width
        self.normalize = normalize
        self.handle_border = handle_border

        self.feature_maps = None
        self.mean_attention_map = None

    def get_mean_attention_map(self, frame):

        height, width = frame.shape[:2]

        self.mean_attention_map = np.zeros((height, width), np.float64)

        frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        self.feature_maps = cv2.split(frame_lab)

        self._generate_mean_attention_map(self.sample_step_size)

        return self.mean_attention_map

    def _generate_mean_attention_map(self, step):
        """Generates a mean attention map

        Args:
            step: The step size to take when going through the min-max interval of feature map values.

        Returns:
            Nothing
        """
        for f in self.feature_maps:
            # To assign limits for thresholding
            min_val, max_val = cv2.minMaxLoc(f)[:2]

            for thresh in range(int(min_val), int(max_val), step):
                # Assign 1 to a pixel if greater than threshold otherwise 0
                bm = cv2.threshold(f, thresh, 1, cv2.THRESH_BINARY)[1]
                bm_opened = self._apply_open(bm)
                self.mean_attention_map += self._get_attention_map(bm_opened)
                # Generate inverse boolean map
                bm = cv2.bitwise_not(bm)
                bm_opened = self._apply_open(bm)
                self.mean_attention_map += self._get_attention_map(bm_opened)

    def _apply_open(self, bm):
        """Applies an opening operation on the raw boolean map

        Args:
            bm: A boolean map

        Returns:
            If opening width > 0, then a new boolean map with opening operation applied, else the same boolean map as input (not a copy)
        """
        bm_opened = None
        if self.opening_width > 0:
            bm_opened = cv2.morphologyEx(bm, cv2.MORPH_OPEN, None, bm_opened, (1,-1), self.opening_width)
        if bm_opened is None:
            return bm
        else:
            return bm_opened

    def _get_attention_map(self, bm):
        am = bm.copy()
        h, w = am.shape[:2]

        # Mask out all the pixels connected to borders
        # since we care only about the pixels that are surrounded
        # and 1 filled regions that are connected to border are
        # not surrounded under the definition

        if self.handle_border:
            for row in range(h):
                # All zero mask means no boundaries for flood filling
                # mask will change after every floodFill by setting the
                # filled pixels to a new value as set in the flags
                # which is 1 by default
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                # Random jump of pixel between 5 and 24 included with probability 0.01, otherwise 0
                jump = random.randint(5, 24) if random.random() > 0.99 else 0
                if am[row, 0+jump] != 1:
                    # Flood fill
                    cv2.floodFill(am, mask, (0+jump, row), (1,), (0,), (0,), 8)
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                # Calculate jump from the other end of the row
                jump = random.randint(5, 24) if random.random() > 0.99 else 0
                if am[row, w-1-jump] != 1:
                    cv2.floodFill(am, mask, (w-1-jump, row), (1,), (0,), (0,), 8)

            for col in range(w):
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                jump = random.randint(5, 24) if random.random() > 0.99 else 0
                if am[0+jump, col] != 1:
                    cv2.floodFill(am, mask, (col, 0+jump), (1,), (0,), (0,), 8)
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                jump = random.randint(5, 24) if random.random() > 0.99 else 0
                if am[h-1-jump, col] != 1:
                    cv2.floodFill(am, mask, (col, h-1-jump), (1,), (0,), (0,), 8)

        else:
            for row in range(h):
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                if am[row, 0] != 1:
                    cv2.floodFill(am, mask, (0, row), (1,), (0,), (0,), 8)
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                if am[row, w-1] != 1:
                    cv2.floodFill(am, mask, (w-1, row), (1,), (0,), (0,), 8)
                for col in range(w):
                    mask = np.zeros((h+2, w+2), dtype=np.uint8)
                if am[0, col] != 1:
                    cv2.floodFill(am, mask, (col, 0), (1,), (0,), (0,), 8)
                mask = np.zeros((h+2, w+2), dtype=np.uint8)
                if am[h-1, col] != 1:
                    cv2.floodFill(am, mask, (col, h-1), (1,), (0,), (0,), 8)

        # Make the pixels not equal to 1 white
        am = (am != 1).astype(np.uint8) * 255

        if self.dilation_width > 0:
            am = cv2.dilate(am, None, am, (-1, -1), self.dilation_width)

        am = am.astype(np.float64)

        if self.normalize:
            cv2.normalize(am, am, 1.0, 0.0, cv2.NORM_L2)
        else:
            cv2.normalize(am, am, 1.0, 0.0, cv2.NORM_MINMAX)

        return am
