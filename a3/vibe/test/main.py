import cv2
import numpy as np
import vibe
import sys

if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit(0)
	
	filename = sys.argv[1]

	cap = cv2.VideoCapture(filename)
	v = vibe.VIBE(3, 20, 4, 17, 2, 16)
	
	ret, frame = cap.read()
	cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
	mframe = vibe.Mat.from_array(frame)
	
	v.init(mframe)
	
	gray = mframe.clone()

	while True:
		v.update(gray)
		foreground = np.asarray(v.getMask())
		cv2.imshow("frame", frame)

		cv2.imshow("foreground", foreground)

		ret, frame = cap.read()
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
		gray = vibe.Mat.from_array(frame)
	
		key = cv2.waitKey(10)
		if key == 27:
			break

	
			
	
