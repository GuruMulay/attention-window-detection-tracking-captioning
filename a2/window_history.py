#!/usr/bin/env python

_windows = []
_sigmas = []
_overlap_limit = 0.5
_window_size_limit = 30


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

def _does_overlap(window, sigma):
    """Checks if the window overlaps with any of previous windows at same sigma

    Args:
        window: a 4-tuple representing attention window rectangle
        sigma: the scale at which attention window was received

    Returns:
        True if window overlaps, else False
    """
    for w, s in zip(_windows, _sigmas):
        if s == sigma and _get_overlap(window, w) > _overlap_limit:
            return True
    return False


def add_if_new(window, sigma):
    """Adds the window to the history if it doesn't overlap with those already present there.
    
    Args:
        window: a 4-tuple representing top-left and bottom-right corners of the attention window.
    Returns:
        True if the window was added, else False
    
    """
    assert len(window) == 4

    is_added = False
    if not _does_overlap(window, sigma):
        _windows.append(window)
        _sigmas.append(sigma)
        is_added = True
    
    # Remove the oldest window
    if len(_windows) > _window_size_limit:
        _windows.pop(0)
        _sigmas.pop(0)

    assert len(_windows) == len(_sigmas)

    return is_added
