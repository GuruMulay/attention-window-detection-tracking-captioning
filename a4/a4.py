import cv2
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize,imshow


import csv
import shapely.geometry as gs

import sys
sys.path.append('vgg')
sys.path.append('../a2')
sys.path.append('../a3')
from vgg16 import vgg16
from imagenet_classes import class_names
from a2 import FeatureDetector
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
    #print len(video_frames)
    cap.release()
    
    
    unionOfForegroundRectsPerFrame = {}
    
    track_dict = {}
    with open('tracks.csv', 'rb') as csvfile:        
        tracks = csv.reader(csvfile, delimiter=',', quotechar='|')            
        if tracks:
            for row in tracks:
                frame_no = int(row[1])
                x1,y1,x2,y2 = row[2:]
                x1 = int(float(x1))
                y1 = int(float(y1))
                x2 = int(float(x2))
                y2 = int(float(y2))
                # let's grab a larger image; add padding
                factor = 0.3
                x1 = max(0,int(x1-(x2-x1)*factor))
                x2 = min(video_frames[frame_no].shape[1]-1,int(x2+(x2-x1)*factor))
                y1 = max(0,int(y1-(y2-y1)*factor))
                y2 = min(video_frames[frame_no].shape[0]-1,int(y2+(y2-y1)*factor))
                
                img = video_frames[int(frame_no)][y1:y2,x1:x2]
                img = resize_image(img)      
                #cv2.imshow(row[0],img)
                #cv2.waitKey(10)
                if row[0] in track_dict:
                    track_dict[row[0]]['frame_end'] = max(track_dict[row[0]]['frame_end'],frame_no)
                    track_dict[row[0]]['image_stack'].append(img)
                else:
                    row_dict = {}
                    row_dict['frame_start'] = frame_no
                    row_dict['frame_end'] = frame_no
                    row_dict['x1'] = int(float(row[2]))
                    row_dict['y1'] = int(float(row[3]))
                    row_dict['image_stack'] = [img]
                    track_dict[row[0]] = row_dict
                
                b = gs.box(x1,y1,x2,y2)
                if frame_no in unionOfForegroundRectsPerFrame:                    
                    unionOfForegroundRectsPerFrame[frame_no] = unionOfForegroundRectsPerFrame[frame_no].union(b)
                else:
                    unionOfForegroundRectsPerFrame[frame_no] = b
                    
                    
    print unionOfForegroundRectsPerFrame
    
    
            
    
    sess = tf.Session()
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(images, 'vgg/vgg16_weights.npz', sess)

    # Add moving objects
    
    for key in track_dict:
        print('object :' + key)
        classified_as = {}
        for image in track_dict[key]['image_stack']:
            cv2.imshow(key,image)
            cv2.waitKey(10)
            image_stack  = np.stack([image])
            probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})            
            preds = np.argmax(probs, axis=1)

            for index, p in enumerate(preds):
                print "Prediction: %s; Probability: %f"%(class_names[p], probs[index, p])
                if class_names[p] in classified_as:
                    classified_as[class_names[p]] += 1
                else:
                    classified_as[class_names[p]] = 1
        most_common_class = ""
        frequency = -1
        for k in classified_as:
            if classified_as[k] > frequency:
                print(classified_as[k], frequency)
                frequency = classified_as[k]
                most_common_class = k                
        print classified_as
        print("Most Common Class: " + most_common_class)
        track_dict[key]['image_stack'] = None
        track_dict[key]['class'] = most_common_class
        track_dict[key]['frequency'] = frequency
    print track_dict
    
    with open('classes.csv', 'wb') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key in track_dict:
            track = track_dict[key]
            frequency = round(track['frequency'] * 100.0 /(track['frame_end'] - track['frame_start']), 2)
            csvWriter.writerow(['moving',track['frame_start'], track['frame_end'], track['x1'], track['y1'], track['class'], frequency])
    
    #Add background features
    print "starting"
    feature_windows = []
    featureDetector = FeatureDetector('dbscan')
    frame_count = 0
    for frame in video_frames:
        if frame_count in unionOfForegroundRectsPerFrame:
            window, aw = featureDetector.get_window(frame,unionOfForegroundRectsPerFrame[frame_count])
        else:
            window, aw = featureDetector.get_window(frame)
        window = resize_image(window)        
        cv2.imshow('feature',window)
        cv2.waitKey(10)
        image_stack  = np.stack([window])
        probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})            
        preds = np.argmax(probs, axis=1)
        for index, p in enumerate(preds):
            print "Prediction: %s; Probability: %f"%(class_names[p], probs[index, p])  
            w = {'frame_number':frame_count, 'x1': aw[0], 'y1': aw[1], 'class': class_names[p], 'activation': probs[index, p],'window':window}
            feature_windows.append(w)    
        frame_count += 1
    
    # sort feature_windows according to activation
    feature_windows = sorted(feature_windows, key=lambda k:k['activation'], reverse=True)
    
    # write best five objects according to activation into the csv file 
    already_found_classes = {}
    with open('classes.csv', 'a') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
        count = 0
        for obj in feature_windows:
            if count < 5:
                if obj['class'] not in already_found_classes:
                    activation = round(obj['activation'],2)
                    frame_number = int(obj['frame_number'])
                    csvWriter.writerow(['still',frame_number, frame_number, int(obj['x1']), int(obj['y1']), obj['class'], activation])
                    cv2.imwrite('img/' + str(count) + '-' + str(activation) + '.jpg', obj['window'])
                    already_found_classes[obj['class']] = True
                    count += 1
            else:
                break
        
