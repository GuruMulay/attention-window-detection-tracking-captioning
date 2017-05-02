from saliency import bms
import cv2

def get_attention_window(frame):
    pass

def get_saliency_map(frame, b, larger_dim=None):
    if larger_dim != None:
        h, w, c = frame.shape
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

    b = None
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if not b:
                b = bms.BMS(frame)
            else:
                b.refresh(frame)
        else:
            break

        cv2.imshow('hello', get_saliency_map(frame, b, 800))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
