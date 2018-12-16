import vibel
import numpy as np


class VIBE:

    def __init__(self, channels=1, samples=20, pixel_neighbor=1, distance_threshold=20, matching_threshold=3, update_factor=16):
        self.instance = vibel.VIBE(channels, samples, pixel_neighbor, distance_threshold, matching_threshold, update_factor)

    def initialize(self, frame):
        frame_m = vibel.Mat.from_array(frame)
        self.instance.init(frame_m)

    def update(self, frame):
        frame_m = vibel.Mat.from_array(frame)
        self.instance.update(frame_m)

    def get_mask(self):
        frame_m = self.instance.getMask()
        return np.array(frame_m)