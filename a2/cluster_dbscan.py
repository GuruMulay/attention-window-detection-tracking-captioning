#!/usr/bin/env python
import cv2
from operator import attrgetter
import window_history
import collections
from pprint import pprint
from sklearn.cluster import DBSCAN
import operator
import numpy as np
import sys

# for coloring different clusters, used modulo when number of clusters is higher than defined colors
colors=[
[0,0,0],
[255,192,170],
[255,0,0],
[0,255,0],
[0,0,255],
[255,255,0],
[0,255,255],
[255,0,255],
[192,192,192],
[128,128,128],
[128,0,0],
[128,128,0],
[0,128,0],
[128,0,128],
[0,128,128],
[0,0,128],
[205,92,92],
[255,160,122],
[255,140,0],
[128,128,0],
[218,165,32],
[173,255,47],
[32,178,170],
]


def cluster_to_window(kps):
	"""Get an attention window from a cluster of keypoints

    Args:
        k: Cluster of keypoints

    Returns:
        Window in the form of a 4-tuple for x,y coordinates for top-left and bottom right corners
    """
	xmax,xmin, ymax, ymin = (-1, 100000,-1, 100000)
	
	for kp in kps:
		x, y = kp.pt
		r = kp.size/2
		if x-r < xmin:
			xmin = x - r
		if x+r > xmax:
			xmax = x + r
		if y-r < ymin:
			ymin = y - r
		if y+r > ymax:
			ymax = y + r
	return (xmin, ymin, xmax, ymax)
    
def get_keypoint_attrs(k):
    """Extracts individual SIFT keypoint details from its octave attribute

    Args:
        k: A keypoint (singular) yielded by SIFT

    Returns:
        (octave, layer, scale): where octave, layer, and scale have usual meanings
    """
    
    octave = k.octave & 255
    layer = (k.octave >> 8) & 255
    octave = octave if octave < 128 else (-128 | octave) # http://code.opencv.org/issues/2987
    scale = 1 / float(1 << octave) if octave >= 0 else (float)(1 << -octave)

    return (octave, layer, scale)   
    
def crop(frame, pt, size):
	'''Get a patch of image around the keypoint, size of the patch is defined the argument "size"
	
	Args:
		frame: full frame image
		pt: coordinate of the keypoint
		size: patch size
		
	Returns:
		a patch of image with dimension (2*size+1, 2*size+1)
	'''
	x0 = int(pt[1]-size)
	if x0 < 0: 
		x0 = 0
			
	x1 = int(pt[1]+size)
	if x1 > frame.shape[0]-1:
		x1 = frame.shape[0]-1
		
	y0 = int(pt[0]-size)
	if y0 < 0:
		y0 = 0

	y1 = int(pt[0]+size)
	if y1 > frame.shape[1]-1:
		y1 = frame.shape[1]-1	
	return frame[x0:x1, y0:y1]

def cluster(frame, sift):
	#frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	kps = sift.detect(frame, None)
	kps.sort(key=attrgetter('octave'), reverse=False)
	
	# group keypoints from different scale to cluster them seperately
	groups = collections.defaultdict(list)
	for kp in kps:
		groups[get_keypoint_attrs(kp)[2]].append(kp)
	
	# for each of the groups get multiple clusters(list of keypoints)
	clusters = []
	avg_response_of_clusters = []
	for item in groups.items():
		#print(str(item[0]) +": "+ str(len(item[1])) + "\n")
		#build cluster		
		X = []
		kp_index = 0;
		
		for kp in item[1]:			
			#creating histogram
			region = crop(frame, kp.pt, 5)
			bgr_hist = []
			histb = cv2.calcHist([region],[0],None, [64], [0,256])
			histg = cv2.calcHist([region],[1],None, [64], [0,256])
			histr = cv2.calcHist([region],[2],None, [64], [0,256])	
			bgr_hist.append(histb)		
			bgr_hist.append(histg)
			bgr_hist.append(histr)			
			a = np.array(bgr_hist)			
			bgr_hist = a.flatten()
			bgr_hist_max = max(bgr_hist)
			bgr_hist_norm = [float(i)/bgr_hist_max for i in bgr_hist] 
			#print(bgr_hist_norm)
			
			#print(bgr_hist)
			# normalize weight and add histogram as feature
			x = [kp.pt[0]/frame.shape[1], kp.pt[1]/frame.shape[0]]
			x = [i*100 for i in x]			
			x += bgr_hist_norm    #adding histogram as features with position
			X.append(x)
			kp.class_id = kp_index
			kp_index = kp_index + 1
			#pprint(dir(kp))
		if item[0] == 2.0:
			db = DBSCAN(eps=3,min_samples=15).fit(X)
		elif item[0] == 1.0:
			db = DBSCAN(eps=5,min_samples=10).fit(X)
		elif item[0] == 0.5:
			db = DBSCAN(eps=7,min_samples=4).fit(X)
		else:
			db = DBSCAN(eps=10,min_samples=2).fit(X)
		labels = db.labels_		
		# Number of clusters in labels, ignoring noise if present
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		
		# assigning corresponding cluster id to the keypoints
		for kp in item[1]:
			kp.class_id = labels[kp.class_id]
		
		# calculate average response of each cluster
		for clstr_no in list(range(0,n_clusters_)):
			clstr = [kp for kp in item[1] if kp.class_id == clstr_no]
			avg_response = np.average([k.response for k in clstr])
			clusters.append(clstr)
			avg_response_of_clusters.append(avg_response)
		
		# Parallel sorting of clusters using their avg. response
		if n_clusters_ > 0:
			avg_response_of_clusters, clusters = zip(*sorted(zip(avg_response_of_clusters, clusters), reverse = True))
			avg_response_of_clusters = list(avg_response_of_clusters)
			clusters = list(clusters)
			#print(clusters)
			#print(avg_response_of_clusters)
		
	#quit()
	
	best_keypoints = []
	frame_attention_window = None
	
	# find best avaiable cluster
	for c in clusters:
		aw = cluster_to_window(c)
		octave, layer, scale= get_keypoint_attrs(c[0])
		if window_history.add_if_new(aw, scale):
			frame_attention_window = aw
			if len(sys.argv) > 3:
				best_keypoints += kps # returning all keypoints for visualization
			else:
				best_keypoints += c # returning only the best cluster
			break;
		
	'''
	for i in range(len(kps)):
		aw = keypoint_to_window(kps[i])
		octave, layer, scale= get_keypoint_attrs(kps[i])
		kp = kps[i]
		#cv2.imshow("Test1", frame[int(kp.pt[1]-5):int(kp.pt[1]+5), int(kp.pt[0]-5):int(kp.pt[0]+5)])
		if window_history.add_if_new(aw, scale):
			frame_attention_window = aw
			best_keypoints += groups[2.0]
			print(scale, groups[scale][3].class_id)
			break
	'''
	return (frame_attention_window, best_keypoints)


