import numpy as np
import shapely.geometry as gs
import csv
import matplotlib.pyplot as plt
import seaborn as sns

######################### change resolution, nFrames ###############################
#resolution 1280x720
img = gs.box(0, 0, 1280, 720)
nFrames = 219

output_PATH = "tracks.csv"
gTruth1_PATH = "ground_truth_partial_car_(1).csv"
gTruth2_PATH = "ground_truth_partial_car_(2).csv"

tracks0 = {}
tracks1 = {}
gt0 = {}
gt1 = {}


gtDict = {}  # ground truth
sutDict = {} # system under test
gtPathList = [gTruth1_PATH, gTruth2_PATH]

def plotOverlap(totalGroundTruthFrames, overlapInPerFrame, overlapOutPerFrame):
    print len(totalGroundTruthFrames), (totalGroundTruthFrames), len(overlapInPerFrame), (overlapInPerFrame)
    sns.set_style("darkgrid")
    inner, = plt.plot(totalGroundTruthFrames, overlapInPerFrame, 'b')
    outer, = plt.plot(totalGroundTruthFrames, overlapOutPerFrame, 'y')
    diff, = plt.plot(totalGroundTruthFrames, overlapInPerFrame-overlapOutPerFrame, 'k')

    plt.legend([inner, outer, diff], ['Inner Overlap', 'Outer Overlap', 'Their Difference'])
    plt.xlabel('Frame Number')
    plt.ylabel('Area Overlap Ratio')
    plt.title('Window Overlap: Ground Truth & Actual Output')

    plt.text(157, 0.025, "mean overlap(GT^Output) = " + str(np.around(np.mean(overlapInPerFrame-overlapOutPerFrame), decimals=4)), fontsize=12)
    plt.show()


def evalMetrics():
    # for output: create the dictionary with frame number as a key and tracker window as values (x1, y1, x2, y2, TrackIndex)
    f = open(output_PATH, 'rt')
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        # print i, row[0]
        row = [float(r) for r in row]
        if row[0]==0:  # TrackIndex == 0
            if row[1] in tracks0.keys():
                print "present------------"
                tracks0[row[1]] += [row[2], row[3], row[4], row[5], row[0]]  # stores outputs x1, y1, x2, y2, TrackIndex
            else:
                tracks0[row[1]] = []
                # print "initialized"
                tracks0[row[1]] += [row[2], row[3], row[4], row[5], row[0]]

        if row[0]==1:  # TrackIndex == 1
            if row[1] in tracks1.keys():
                print "present------------"
                tracks1[row[1]] += [row[2], row[3], row[4], row[5], row[0]]  # stores outputs x1, y1, x2, y2, TrackIndex
            else:
                tracks1[row[1]] = []
                # print "initialized"
                tracks1[row[1]] += [row[2], row[3], row[4], row[5], row[0]]

    print "tracks0 dict", len(tracks0), tracks0
    print "tracks1 dict", len(tracks1), tracks1


    # create the dictionary with frame number as a key and tracker window as values (x1, y1, x2, y2)
    f = open(gTruth1_PATH, 'rt')
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for i, row in enumerate(reader):
        # print i, row
        row = [float(r) for r in row]
        if row[0] in gt0.keys():
            print "present------------"
            gt0[row[0]] += [row[1], row[2], row[7], row[8]]  # stores ground truth x1, y1, x2, y2
        else:
            gt0[row[0]] = []
            # print "initialized"
            gt0[row[0]] += [row[1], row[2], row[7], row[8]]

    print "gt0 dict", len(gt0), gt0

    # create the dictionary with frame number as a key and tracker window as values (x1, y1, x2, y2)
    f = open(gTruth2_PATH, 'rt')
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for i, row in enumerate(reader):
        # print i, row
        row = [float(r) for r in row]
        if row[0] in gt1.keys():
            print "present------------"
            gt1[row[0]] += [row[1], row[2], row[7], row[8]]  # stores ground truth x1, y1, x2, y2
        else:
            gt1[row[0]] = []
            # print "initialized"
            gt1[row[0]] += [row[1], row[2], row[7], row[8]]

    print "gt1 dict", len(gt1), gt1


    # overlap calculations
    overlap0 = 0
    overlap0Th = 0.1  # set required threshold of overlap

    keysetIntersect = np.intersect1d(tracks0.keys(), gt0.keys())
    keysetUnion = np.union1d(tracks0.keys(), gt0.keys())
    keysetIntersect = keysetIntersect.astype(np.float)
    keysetUnion = keysetUnion.astype(np.float)

    keysetIntersect_fail = 0

    # truePositivePerFrame = []
    truePositiveAreaPerFrame = np.zeros(len(keysetUnion))
    falsePositiveAreaPerFrame = np.zeros(len(keysetUnion))
    # trueNegativePerFrame = np.zeros(len(keysetUnion))
    # falseNegativePerFrame = np.zeros(len(keysetUnion))

    print "keys length", len(keysetIntersect)
    keyMin = np.min(keysetUnion)
    for i, key in enumerate(keysetIntersect):
        # if key not in gt0.keys() or key not in tracks0.keys():
        out = gs.box(tracks0[key][0], tracks0[key][1], tracks0[key][2], tracks0[key][3])
        gt = gs.box(gt0[key][0], gt0[key][1], gt0[key][2], gt0[key][3])
        gtIntOut_area = out.intersection(gt).area
        area_ratio = gtIntOut_area/gt.area   # common area w.r.t. ground truth area
        truePositiveAreaPerFrame[int(key)-int(keyMin)] = area_ratio
        outer_area = img.area-gt.area
        outer_ratio = (out.area-gtIntOut_area)/outer_area
        falsePositiveAreaPerFrame[int(key)-int(keyMin)] = outer_ratio
        print "iter, key(frameNo), valOut, valGT, area, overlapIn, overlapOut:   ", i, key, tracks0[key], gt0[key], gtIntOut_area, area_ratio, outer_ratio

        # if area_ratio < overlap0Th:  # threshold for true detection
        #     keysetIntersect_fail += 1
        #     continue  # don't accumulate in sum of overlaps
        overlap0 = overlap0 + area_ratio

    print "i, overlap0, len(keysetIntersect) out of len(keysetUnion)", i, overlap0/i, len(keysetIntersect)-keysetIntersect_fail, len(keysetUnion)
    # print "overlapPerFrame", len(truePositiveAreaPerFrame), truePositiveAreaPerFrame
    plotOverlap(keysetUnion, truePositiveAreaPerFrame, falsePositiveAreaPerFrame)



    overlap1 = 0
    overlap1Th = 0.1  # set required threshold of overlap

    keysetIntersect = np.intersect1d(tracks1.keys(), gt1.keys())
    keysetUnion = np.union1d(tracks1.keys(), gt1.keys())
    keysetIntersect = keysetIntersect.astype(np.float)
    keysetUnion = keysetUnion.astype(np.float)

    keysetIntersect_fail = 0
    truePositiveAreaPerFrame = np.zeros(len(keysetUnion))
    falsePositiveAreaPerFrame = np.zeros(len(keysetUnion))

    print "keys length", len(keysetIntersect)
    keyMin = np.min(keysetUnion)
    for i, key in enumerate(keysetIntersect):
        # if key not in gt1.keys() or key not in tracks1.keys():
        out = gs.box(tracks1[key][0], tracks1[key][1], tracks1[key][2], tracks1[key][3])
        gt = gs.box(gt1[key][0], gt1[key][1], gt1[key][2], gt1[key][3])
        gtIntOut_area = out.intersection(gt).area
        area_ratio = gtIntOut_area/gt.area   # common area w.r.t. ground truth area
        truePositiveAreaPerFrame[int(key) - int(keyMin)] = area_ratio
        outer_area = img.area - gt.area
        outer_ratio = (out.area - gtIntOut_area)/outer_area
        falsePositiveAreaPerFrame[int(key) - int(keyMin)] = outer_ratio
        print "iter, key(frameNo), valOut, valGT, area, overlapIn, overlapOut:   ", i, key, tracks1[key], gt1[key], gtIntOut_area, area_ratio, outer_ratio


        # if area_ratio < overlap1Th:  # threshold for true detection
        #     keysetIntersect_fail += 1
        #     continue   # don't accumulate in sum of overlaps
        overlap1 = overlap1 + area_ratio

    print "i, overlap1, len(keysetIntersect) out of len(keysetUnion)", i, overlap1/i, len(keysetIntersect)-keysetIntersect_fail, len(keysetUnion)
    plotOverlap(keysetUnion, truePositiveAreaPerFrame, falsePositiveAreaPerFrame)


def evaluation():
    # for output: create the dictionary with frame number as a key and tracker window as values (x1, y1, x2, y2, TrackIndex)
    f = open(output_PATH, 'rt')
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        # print i, row[0]
        row = [float(r) for r in row]
        if [int(row[1]), int(row[0])] in sutDict.keys(): # print "present------------"
            sutDict[int(row[1]), int(row[0])] += [row[2], row[3], row[4], row[5]]  # stores outputs x1, y1, x2, y2, TrackIndex
        else:
            sutDict[int(row[1]), int(row[0])] = [] # print "initialized"
            sutDict[int(row[1]), int(row[0])] += [row[2], row[3], row[4], row[5]]

    print "sutDict", len(sutDict), sutDict

    # create the dictionary with frame number as a key and tracker window as values (x1, y1, x2, y2)
    for n, gt in enumerate(gtPathList):
        f = open(gt, 'rt')
        # print np.indexof(gt)
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        for i, row in enumerate(reader):
            # print i, row
            row = [float(r) for r in row]
            if [int(row[0]), n] in gtDict.keys():  # print "present------------"
                gtDict[int(row[0]), n] += [row[1], row[2], row[7], row[8]]  # stores ground truth x1, y1, x2, y2
            else:
                gtDict[int(row[0]), n] = []  # print "initialized"
                gtDict[int(row[0]), n] += [row[1], row[2], row[7], row[8]]

    print "gtDict dict", len(gtDict), gtDict

def getOverlapRatio(vs, vg):
    out = gs.box(vs[0], vs[1], vs[2], vs[3])
    gt = gs.box(vg[0], vg[1], vg[2], vg[3])
    gtIntOut_area = out.intersection(gt).area
    area_ratio = gtIntOut_area / gt.area
    return area_ratio

def frameBasedMetric():
    tp_th = 0.25
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    gtKeysFrames = np.array([k[0] for k in gtDict.iterkeys()])  # [[k[0], k[1]] for k in gtDict.iterkeys()]
    gtKeysTrackId = np.array([k[1] for k in gtDict.iterkeys()])
    sutKeysFrames = np.array([k[0] for k in sutDict.iterkeys()])  # [[k[0], k[1]] for k in sutDict.iterkeys()]
    sutKeysTrackId = np.array([k[1] for k in sutDict.iterkeys()])
    # gtKeysFrames = np.array(gtKeysFrames)
    # sutKeysFrames = np.array(sutKeysFrames)

    print "original keys, frames, tracks", len([k for k in gtDict.iterkeys()]), len(gtKeysFrames), len(gtKeysTrackId)
    print "original keys, frames, tracks", len([k for k in sutDict.iterkeys()]), len(sutKeysFrames), len(sutKeysTrackId)
    # print sutKeysFrames
    # print sutKeysTrackId
    # print np.sort(sutKeysFrames)

    for frNo in range(nFrames):
        # TN
        if (frNo not in gtKeysFrames) and (frNo not in sutKeysFrames):
            tn += 1
            print "TN", frNo
            continue

        # TP
        if (frNo in gtKeysFrames) and (frNo in sutKeysFrames):
            gtFrame = dict((key, value) for key, value in gtDict.iteritems() if key[0] == frNo)
            # print "Ground truth", gtFrame
            sutFrame = dict((key, value) for key, value in sutDict.iteritems() if key[0] == frNo)
            # print "System out::", sutFrame

            # if len(sutFrame) < len(gtFrame): # if detected windows are lesser than gt
            #     continue
            cnt = 0
            for ks, vs in sutFrame.iteritems():
                # print ks, vs
                for kg, vg in gtFrame.iteritems():
                    # print kg, vg
                    overlapR = getOverlapRatio(vs, vg)
                    # print "overlapR", overlapR
                    if overlapR>tp_th:  # only one overlap >th is sufficient (according to paper [3])
                        tp += 1
                        break
                    else:
                        cnt +=1

                if overlapR > tp_th:  # only one overlap >th is sufficient (according to paper [3])
                    break

            # print cnt, len(sutFrame)*len(gtFrame)
            if cnt==len(sutFrame)*len(gtFrame):
                fn += 1
                # print "fn -------", fn
            print "TP", frNo
            continue  # for if loop of tp

        # FN
        if (frNo in gtKeysFrames) and (frNo not in sutKeysFrames):
            # print frNo
            fn += 1
            print "FN", frNo
            continue

        # FP
        if (frNo not in gtKeysFrames) and (frNo in sutKeysFrames):
            fp += 1
            print "FP", frNo
            continue

    print "tn, tp, fn, fp", tn, tp, fn, fp
    plotMeasures(tn, tp, fn, fp)

    # print len(np.intersect1d(gtKeysFrames,sutKeysFrames))  #, np.sort(gtKeysFrames), np.sort(sutKeysFrames)
    # print len(np.union1d(gtKeysFrames,sutKeysFrames))




    # print gtDict[(100,any())]
    # for k in gtDict.iterkeys():
    #     if k[0] == 100:
    #         print k

    # print frNo, gtDict[frNo,0], sutDict[frNo,0]
    # print np.where(gtKeysFrames == frNo)
    # gt_tracks = [i for i in np.where(gtKeysFrames == frNo)[0]]
    # print gt_tracks

def plotMeasures(tn, tp, fn, fp):
    ### Measures ------------------------------------------
    FAR = float(fp)/float(tp+fp)
    print "False Alarm Rate", FAR

    DetectionRate = float(tp)/float(tp+fn)
    print "Detection Rate", DetectionRate

    # Accuracy = (tp+tn)/tf
    False_Negative_Rate = float(fn)/float(fn+tp)
    False_Positive_Rate = float(fp)/float(fp+tn)

    print "False_Negative_Rate ", False_Negative_Rate
    print "False_Positive_Rate ", False_Positive_Rate

    precision = float(tp)/float(tp+fp)
    recall = float(tp)/float(tp+fn)

    print "precision", precision
    print "recall", recall

if __name__ == '__main__':
    # evalMetrics()

    evaluation()
    frameBasedMetric()

    # plotMeasures(63, 139, 17, 0)

    # keysetUnion = keysetUnion.astype(np.float)
    # for i, key in enumerate(np.sort(keysetUnion)):
    #     print "Union", i, key

    # # ----------------------------------------- testing
    # x1,y1,x2,y2 = [1,1,11,11]
    # x11,y11,x21,y21 = [0,0,16,16]
    #
    # a = gs.box(x1,y1,x2,y2)
    # b = gs.box(x11,y11,x21,y21)
    #
    # print(a.area), b.contains(a.centroid), a.centroid
    # print(b.area)
    #
    # gtIntOut_area = a.intersection(b).area
    # print(gtIntOut_area)