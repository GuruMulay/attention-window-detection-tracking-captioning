import bmsl
import numpy as np

class BMS:
    """
    Computes Saliency map using Boolean map based saliency approach as described in:
    "Saliency Detection: A Boolean Map Approach", Jianming Zhang,
    Stan Sclaroff, ICCV, 2013
    """
    def __init__(self, src, dilation_width=7, opening_width=5, normalize=True, handle_borders=True):
        """
        Initialize the Boolean Map Saliency object
        :param src: image array whose saliency map is to be computed
        :param dilation_width: kernel width of the dilation operation in pixels
        :param opening_width: kernel width of the opening operation in pixels
        :param normalize: enable normalization emaphasize attention maps with small active areas
        :param handle_borders: enable handling of borders
        """
        self.dw = dilation_width
        self.ow = opening_width
        self.normalize = normalize
        self.hb = handle_borders

        self.instance = bmsl.BMS(bmsl.Mat.from_array(src), dilation_width, opening_width, normalize, handle_borders)

    def get_saliency_map(self, step_size=8):
        """
        Returns a Salincy map of the image
        :param step_size: step size for sampling the threshold between 0 to 255
        :return: saliency map image
        """
        self.instance.computeSaliency(step_size)
        out_m = self.instance.getSaliencyMap()
        return np.array(out_m)

    def refresh(self, src, dilation_width=None, opening_width=None, normalize=None, handle_borders=None):
        """
        Update the source image, and optionally the algorithm parameters
        :param src: image array whose saliency map is to be computed
        :param dilation_width: kernel width of the dilation operation in pixels
        :param opening_width: kernel width of the opening operation in pixels
        :param normalize: enable normalization emaphasize attention maps with small active areas
        :param handle_borders: enable handling of borders
        :return: None, use get_saliency_map to get the saliency map for the new image
        """

        # Use the parameters as passed to initial constructor unless explicitly updated here
        if not dilation_width:
            dilation_width = self.dw
        if not opening_width:
            opening_width = self.ow
        if not normalize:
            normalize = self.normalize
        if not handle_borders:
            handle_borders = self.hb
        self.instance = bmsl.BMS(bmsl.Mat.from_array(src), dilation_width, opening_width, normalize, handle_borders)