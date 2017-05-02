from saliency import bms
import cv2
import numpy as np


def get_saliency_map(frame, b, larger_dim=None):
    if larger_dim != None:
        h, w, _ = frame.shape
        aspect_ratio = w / float(h)
        # Make the bigger dimension equal to larger_dim
        new_dim = (int(larger_dim * aspect_ratio), int(larger_dim)) if w <= h else (int(larger_dim), int(larger_dim/aspect_ratio))
        small_frame = cv2.resize(frame, new_dim)
        b.refresh(small_frame)
        small_sm = b.get_saliency_map()
        sm = cv2.resize(small_sm, (w, h))
    else:
        b.refresh(frame)
        sm = b.get_saliency_map()
    return sm

if __name__ == '__main__':
    vid = 'example1.mov'
    cap = cv2.VideoCapture(vid)

    gsm = None
    b = None
    fnum = 0
    att_h = None
    att_w = None
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if not b:
                b = bms.BMS(frame)
                gsm = np.zeros(frame.shape[:2])
            else:
                b.refresh(frame)
        else:
            break

        sm = get_saliency_map(frame, b, 600)
        sm_norm = sm / 255.0

        leakage = 0.02
        thresh = 15.0

        # Accumulate new response with leakage
        gsm += sm_norm
        gsm -= leakage * gsm

        fh, fw, _ = frame.shape
        window_size = int(fw / 6.0) if fw <= fh else int(fh / 6.0)

        max_h, max_w = np.unravel_index(gsm.argmax(), gsm.shape)

        if att_h is None and att_w is None:
            att_h, att_w = max_h, max_w
        elif gsm[max_h, max_w] > thresh:
            att_h, att_w = max_h, max_w

            top = max_h - window_size if max_h - window_size >= 0 else 0
            bottom = max_h + window_size if max_h + window_size <= fh else fh
            left = max_w - window_size if max_w - window_size >= 0 else 0
            right = max_w + window_size if max_w + window_size <= fw else fw

            gsm[top:bottom, left:right] = 0.0

            cv2.imshow("Attention Window", frame[top:bottom, left:right])

        sm = cv2.drawMarker(sm, (att_w, att_h), color=255, markerType=cv2.MARKER_SQUARE, markerSize=window_size,
                            thickness=5)
        sm = cv2.putText(sm, str(fnum), (0, fh), fontFace=cv2.FONT_ITALIC, fontScale=1, color=255)

        cv2.imshow("Saliency map", sm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fnum += 1

    cap.release()
    cv2.destroyAllWindows()
