from saliency import bms
import cv2
import numpy as np


class StaticAttentionWindow:
    """
    Attention window computer suited for static frames
    """
    def __init__(self, thresh, leakage=None, larger_dim=None):
        self.thresh = thresh
        self.leakage = leakage
        self.larger_dim = larger_dim
        self.gsm = None
        self.b = None
        self.frame = None
        self.aw_x, self.aw_y = (None, None)
        self.top, self.left, self.bottom, self.right = (None, None, None, None)

    def input(self, frame):
        self.frame = frame
        if not self.b:
            self.b = bms.BMS(self.frame)
            self.gsm = np.zeros(self.frame.shape[:2])
        else:
            self.b.refresh(self.frame)

        sm = self._get_saliency_map()
        sm_norm = sm / 255.0
        # Accumulate new response with leakage
        self.gsm += sm_norm
        if self.leakage:
            # Should account for values getting negative
            self.gsm -= self.leakage * self.gsm
            self.gsm[self.gsm < 0.0] = 0.0
        max_y, max_x = np.unravel_index(self.gsm.argmax(), self.gsm.shape)
        if (self.aw_x is None and self.aw_y is None) or self.gsm[max_y, max_x] > self.thresh:
            self.aw_x, self.aw_y = max_x, max_y

        fh, fw, _ = self.frame.shape
        window_size = int(fw / 6.0) if fw <= fh else int(fh / 6.0)
        self.top, self.left, self.bottom, self.right = self._get_window_around(self.aw_x, self.aw_y, window_size)

        self.gsm[self.top:self.bottom, self.left:self.right] = 0.0

    def _get_saliency_map(self):
        if self.larger_dim != None:
            h, w, _ = self.frame.shape
            aspect_ratio = w / float(h)
            # Make the bigger dimension equal to larger_dim
            new_dim = (int(self.larger_dim * aspect_ratio), int(self.larger_dim)) if w <= h else (
                       int(self.larger_dim), int(self.larger_dim / aspect_ratio))
            small_frame = cv2.resize(self.frame, new_dim)
            self.b.refresh(small_frame)
            small_sm = self.b.get_saliency_map()
            sm = cv2.resize(small_sm, (w, h))
        else:
            self.b.refresh(frame)
            sm = self.b.get_saliency_map()
        return sm

    def _get_window_around(self, x, y, window_size):
        fh, fw, _ = self.frame.shape
        top = max(y - window_size, 0)
        bottom = min(y + window_size, fh)
        left = max(x - window_size, 0)
        right = min(x + window_size, fw)

        return top, left, bottom, right

    def get_attention_coordinates(self):
        return self.aw_x, self.aw_y

    def get_attention_window(self):
        return self.frame[self.top:self.bottom, self.left:self.right]

if __name__ == '__main__':
    vid = 'example1.mov'
    cap = cv2.VideoCapture(vid)
    saw = StaticAttentionWindow(4.0, leakage=0.1, larger_dim=600)

    fnum = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        saw.input(frame)
        aw_x, aw_y = saw.get_attention_coordinates()
        aw = saw.get_attention_window()
        #cv2.imwrite('out/' + str(fnum) + '.png', aw)

        frame = cv2.drawMarker(frame, (aw_x, aw_y), color=255, markerType=cv2.MARKER_SQUARE, markerSize=max(aw.shape),
                            thickness=5)
        frame = cv2.putText(frame, str(fnum), (0, frame.shape[0]), fontFace=cv2.FONT_ITALIC, fontScale=1, color=255)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fnum += 1

    cap.release()
    cv2.destroyAllWindows()
