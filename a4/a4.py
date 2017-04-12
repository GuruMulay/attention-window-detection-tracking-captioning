import cv2
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize,imshow

from PIL import Image
from resizeimage import resizeimage
import csv

import sys
sys.path.append('vgg')
sys.path.append('../a2')
sys.path.append('../a3')
from vgg16 import vgg16
from imagenet_classes import class_names
from a2 import FeatureDetection
from mog_mosse import MotionTracker

def resize_image(img):
    h = 224
    w = 224
    if img.shape[0] < img.shape[1]:
        w = int((224.0 * img.shape[1]) / img.shape[0])
        img1 = imresize(img, (h, w))
        c = w/2
        img1 = img1[:,c-h/2:c+h/2]
    elif img.shape[0] > img.shape[1]:
        h = int((224.0 * img.shape[0]) / img.shape[1])
        img1 = imresize(img, (h, w))
        c = h/2
        img1 = img1[c-w/2:c+w/2:,:]
    else:
        img1 = imresize(img, (h, w))
    
    return img1
    
    


if __name__ == '__main__':
    '''
    fd_img = imread('cow.jpg', mode='RGB')
    img = resize_image(fd_img)
    print img.shape
    imshow(img)
    
    cv2.waitKey()
    '''
    
    try:
        video_src = sys.argv[1]
    except:
        video_src = '0'
        
        
    app = MotionTracker(video_src)
    app.run()    
    app.out.release()
    cap = cv2.VideoCapture(video_src)
    video_frames = []
    while(1):
        ret, frame = cap.read()
        if ret:
            video_frames.append(frame)
        else:
            break;
    print len(video_frames)
    cap.release()
    
    track_dict = {}
    with open('tracks.csv', 'rb') as csvfile:        
        tracks = csv.reader(csvfile, delimiter=',', quotechar='|')            
        if tracks:
            for row in tracks:
                x1,y1,x2,y2 = row[2:]
                x1 = int(float(x1))
                y1 = int(float(y1))
                x2 = int(float(x2))
                y2 = int(float(y2))
                # let's grab a larger image; add padding
                factor = 0.3
                x1 = max(0,int(x1-(x2-x1)*factor))
                x2 = min(video_frames[int(row[1])].shape[1]-1,int(x2+(x2-x1)*factor))
                y1 = max(0,int(y1-(y2-y1)*factor))
                y2 = min(video_frames[int(row[1])].shape[0]-1,int(y2+(y2-y1)*factor))
                
                img = video_frames[int(row[1])][y1:y2,x1:x2]
                img = resize_image(img)      
                cv2.imshow(row[0],img)
                cv2.waitKey(10)
                if row[0] in track_dict:
                    track_dict[row[0]]['frame_end'] = max(track_dict[row[0]]['frame_end'],int(row[1]))
                    track_dict[row[0]]['image_stack'].append(img)
                else:
                    row_dict = {}
                    row_dict['frame_start'] = int(row[1])
                    row_dict['frame_end'] = int(row[1])
                    row_dict['x1'] = int(float(row[2]))
                    row_dict['y1'] = int(float(row[3]))
                    row_dict['image_stack'] = [img]
                    track_dict[row[0]] = row_dict
                #print row[2:]
                #x1,y1,x2,y2 = row[2:]
                #x1 = int(float(x1))
                #y1 = int(float(y1))
                #x2 = int(float(x2))
                #y2 = int(float(y2))
                #img = video_frames[int(row[1])][y1:y2,x1:x2]
                #img = resize_image(img)
                #cv2.imshow(row[0],img)
                #cv2.waitKey(100)
    print track_dict
    
    
            
    
    sess = tf.Session()
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(images, 'vgg/vgg16_weights.npz', sess)


    for key in track_dict:
        print('object :' + key)
        for image in track_dict[key]['image_stack']:    
            cv2.imshow(key,image)
            cv2.waitKey(10)
            image_stack  = np.stack([image])
            probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})            
            preds = np.argmax(probs, axis=1)

            for index, p in enumerate(preds):
                print "Prediction: %s; Probability: %f"%(class_names[p], probs[index, p])
    
