import numpy as np
import shapely.geometry as gs
import csv
import matplotlib.pyplot as plt
import seaborn as sns

output_PATH = "tracks.csv"
gTruth1_PATH = "ground_truth_partial_car_(1).csv"
gTruth2_PATH = "ground_truth_partial_car_(2).csv"

tracks0 = {}
tracks1 = {}
gt0 = {}
gt1 = {}

def plotOverlap(totalGroundTruthFrames, ovelapPerFrame):
    print len(totalGroundTruthFrames), (totalGroundTruthFrames), len(ovelapPerFrame), (ovelapPerFrame)
    sns.set_style("darkgrid")
    plt.plot(totalGroundTruthFrames, ovelapPerFrame)

    plt.xlabel('Frame Number')
    plt.ylabel('Area Overlap Ratio')
    plt.title('Ground Truth INTERSECT Actual Output')

    plt.text(155, 0.05, "mean overlap(GT^Output) = "+ str(np.around(np.mean(ovelapPerFrame), decimals=4)), fontsize=12)
    plt.show()


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

# overlapPerFrame = []
overlapPerFrame = np.zeros(len(keysetUnion))
# print overlapPerFrame

print "keys length", len(keysetIntersect)
keyMin = np.min(keysetUnion)
for i, key in enumerate(keysetIntersect):
    # if key not in gt0.keys() or key not in tracks0.keys():
    a = gs.box(tracks0[key][0], tracks0[key][1], tracks0[key][2], tracks0[key][3])
    b = gs.box(gt0[key][0], gt0[key][1], gt0[key][2], gt0[key][3])
    common_area = a.intersection(b).area
    area_ratio = common_area/b.area   # common area w.r.t. ground truth area
    print "iter, key(frameNo), valOut, valGT, area, overlap:   ", i, key, tracks0[key], gt0[key], common_area, area_ratio
    overlapPerFrame[int(key)-int(keyMin)] = area_ratio
    if area_ratio < overlap0Th:  # threshold for true detection
        keysetIntersect_fail += 1
        continue  # don't accumulate in sum of overlaps
    overlap0 = overlap0 + area_ratio

print "i, overlap0, len(keysetIntersect) out of len(keysetUnion)", i, overlap0/i, len(keysetIntersect)-keysetIntersect_fail, len(keysetUnion)
# print "ovelapPerFrame", len(overlapPerFrame), overlapPerFrame
plotOverlap(keysetUnion, overlapPerFrame)

overlap1 = 0
overlap1Th = 0.1  # set required threshold of overlap

keysetIntersect = np.intersect1d(tracks1.keys(), gt1.keys())
keysetUnion = np.union1d(tracks1.keys(), gt1.keys())
keysetIntersect = keysetIntersect.astype(np.float)
keysetUnion = keysetUnion.astype(np.float)

keysetIntersect_fail = 0
overlapPerFrame = np.zeros(len(keysetUnion))

print "keys length", len(keysetIntersect)
keyMin = np.min(keysetUnion)
for i, key in enumerate(keysetIntersect):
    # if key not in gt1.keys() or key not in tracks1.keys():
    a = gs.box(tracks1[key][0], tracks1[key][1], tracks1[key][2], tracks1[key][3])
    b = gs.box(gt1[key][0], gt1[key][1], gt1[key][2], gt1[key][3])
    common_area = a.intersection(b).area
    area_ratio = common_area/b.area   # common area w.r.t. ground truth area
    print "iter, key(frameNo), valOut, valGT, area, overlap:   ", i, key, tracks1[key], gt1[key], common_area, area_ratio
    overlapPerFrame[int(key) - int(keyMin)] = area_ratio
    if area_ratio < overlap1Th:  # threshold for true detection
        keysetIntersect_fail += 1
        continue   # don't accumulate in sum of ovelaps
    overlap1 = overlap1 + area_ratio

print "i, overlap1, len(keysetIntersect) out of len(keysetUnion)", i, overlap1/i, len(keysetIntersect)-keysetIntersect_fail, len(keysetUnion)
plotOverlap(keysetUnion, overlapPerFrame)


# keysetUnion = keysetUnion.astype(np.float)
# for i, key in enumerate(np.sort(keysetUnion)):
#     print "Union", i, key

# # ----------------------------------------- testing
# x1,y1,x2,y2 = [1,1,11,11]
# x11,y11,x21,y21 = [6,6,16,16]
#
# a = gs.box(x1,y1,x2,y2)
# b = gs.box(x11,y11,x21,y21)
#
# print(a.area)
# print(b.area)
#
# common_area = a.intersection(b).area
# print(common_area)