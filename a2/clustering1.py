import cv2
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import window_history
from random import *
from matplotlib import pyplot as plt


c_dict_kp = {}
c_dict_vals = {}


def get_keypoint_matrix(kp):
    kp_mat = []
    for k in kp:
        # ---------------------------
        octave = k.octave & 255
        layer = (k.octave >> 8) & 255
        octave = octave if octave < 128 else (-128 | octave)  # http://code.opencv.org/issues/2987
        scale = 1 / float(1 << octave) if octave >= 0 else (float)(1 << -octave)
        # ---------------------------
        kp_mat.append([k.pt[0], k.pt[1], k.size, k.response, k.angle, k.class_id, k.octave, octave, layer, scale])
    kp_mat = np.array(kp_mat)
    return kp_mat


def get_keypoint_attrs(k):
    """Extracts individual SIFT keypoint details from its octave attribute

    Args:
        k: A keypoint (singular) yielded by SIFT

    Returns:
        (octave, layer, scale): where octave, layer, and scale have usual meanings
    """

    octave = k.octave & 255
    layer = (k.octave >> 8) & 255
    octave = octave if octave < 128 else (-128 | octave)  # http://code.opencv.org/issues/2987
    scale = 1 / float(1 << octave) if octave >= 0 else (float)(1 << -octave)

    return (octave, layer, scale)


def clustering_pixels(X, n_clusters):
    # print "clustering pixels into", n_clusters, "clusters ..."
    Z = linkage(X, method='complete', metric='euclidean')  # method='single', 'centroid', metric='euclidean'
    # clusters = fcluster(Z, n_clusters, criterion='maxclust')
    clusters = fcluster(Z, t=2.5, depth=10)
    # print "clusters are:", clusters
    return clusters


def cluster_keypoints(frame, sift,unionOfForegroundRects=None):
    # frame = cv2.medianBlur(frame, 5)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.Canny(frame_gray, 100, 200, 7)

    kps = sift.detect(frame_gray, None)
    kps = np.array(kps)
    kp_mat = get_keypoint_matrix(kps)

    kp_select = []
    for i in range(len(kps)): # Conditions for keypoint selection
        if kps[i].size > 0.5 * np.mean(kp_mat[:, [2]]) and kps[i].response > 0.5 * np.mean(kp_mat[:, [3]]):
            # if kp_mat[i][7] > 0 and  kp_mat[i][7] < 7 and kp_mat[i][9] < 3:
            if kp_mat[i][9] < 2:
                kp_select.append(kps[i])
    kp_select = np.array(kp_select)

    # Clustering x, y points ==============================
    kp_mat = get_keypoint_matrix(kp_select)
    kp_xy = np.array(np.concatenate((kp_mat[:, [0]], kp_mat[:, [1]]), 1))
    X = kp_xy
    # print "x, y coordinates shape for kp: ", kp_xy.shape

    # pass the number of clusters to the function
    clusters = clustering_pixels(X, n_clusters=25)
    # print "cluster shape", clusters.shape

    # cluster processing =================
    number_of_clusters = np.max(clusters)

    for i, cid in enumerate(np.array(clusters)):
        if cid in c_dict_kp.keys():
            c_dict_kp[cid] += [kp_select[i]]
        else:
            c_dict_kp[cid] = []
            c_dict_kp[cid] += [kp_select[i]]

    for cn in range(1, number_of_clusters):
        keypoint_list = c_dict_kp[cn][:]

        x_coords = []
        y_coords = []
        k_sizes = []
        k_resp = []
        for i in range(len(keypoint_list)):
            x_coords.append(keypoint_list[i].pt[0])
            y_coords.append(keypoint_list[i].pt[1])
            k_sizes.append(keypoint_list[i].size)
            k_resp.append(keypoint_list[i].response)
            k_resp.append(keypoint_list[i].response)

        c_dict_vals[cn] = []
        c_dict_vals[cn] += [[keypoint_list], [np.mean(k_sizes) * np.mean(k_resp)],
                            [np.median(x_coords), np.median(y_coords), np.std(x_coords), np.std(y_coords), np.mean(k_sizes),
                             np.mean(k_resp)]]

    sorted_c_dict = sorted(c_dict_vals.items(), key=lambda x: x[1][1], reverse=True)  # descending order of size*response

    # n = randrange(10)
    # print "randomly chosen out of top 10 clusters: ", sorted_c_dict[n]  # select random element out of top 10

    for n in range(number_of_clusters-1):
        best_keypoints_list = sorted_c_dict[n][1][0]
        best_keypoints = best_keypoints_list[:][0]

        x1 = sorted_c_dict[n][1][2][0] - 0.7*sorted_c_dict[n][1][2][2]
        x2 = sorted_c_dict[n][1][2][0] + 0.7*sorted_c_dict[n][1][2][2]
        y1 = sorted_c_dict[n][1][2][1] - 0.7*sorted_c_dict[n][1][2][3]
        y2 = sorted_c_dict[n][1][2][1] + 0.7*sorted_c_dict[n][1][2][3]

        aw = [x1, y1, x2, y2]
        # to handle the aw == None case
        if n == 1:
            frame_attention_window = aw

        octave, layer, scale = get_keypoint_attrs(best_keypoints_list[0][0]) # choose the 0th keypoint
        if window_history.add_if_new(aw, scale,unionOfForegroundRects):
            frame_attention_window = aw
            break

    # plot the clusters
    # plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
    # plt.gca().invert_yaxis()
    # plt.show()

    assert frame_attention_window != None
    return (frame_attention_window, best_keypoints)

