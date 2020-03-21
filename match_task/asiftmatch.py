from match_task.common import *
import heapq


class ImagePairAsiftMatches:
    def __init__(self):
        self.asift_feature_matchings = []

    def add_match(self, kpts1, kpts2, matches):
        self.asift_feature_matchings.append((kpts1, kpts2, matches))

    def getAll(self):
        kp1_all = []
        kp2_all = []
        matches_all = []
        kp1_next = 0
        kp2_next = 0
        for i in range(0, len(self.asift_feature_matchings)):
            for match in self.asift_feature_matchings[i][2]:
                kp1_all.append(self.asift_feature_matchings[i][0][match.queryIdx])
                match.queryIdx = kp1_next
                kp1_next += 1
                kp2_all.append(self.asift_feature_matchings[i][1][match.trainIdx])
                match.trainIdx = kp2_next
                kp2_next += 1
                matches_all.append(match)
        return kp1_all, kp2_all, matches_all


class SingleViewpointAsiftFeatures:
    def __init__(self, tilt, phi, sift_points, sift_desc):
        self.tilt = tilt
        self.phi = phi
        self.sift_points = sift_points
        self.sift_desc = sift_desc

    def __init__(self, tilt, phi):
        self.tilt = tilt
        self.phi = phi
        self.sift_points = []
        self.sift_desc = []

    def normalize(self, rotmat, tiltmat):
        constant = np.array([[0, 0, 1]])
        affinemat = np.matmul(np.concatenate([tiltmat, constant]), np.concatenate([rotmat, constant]))[:-1]
        inversemat = cv.invertAffineTransform(affinemat)
        self._inverse_tilt_rot(inversemat)

    def _inverse_tilt_rot(self, inversewmat):
        # 2 by 3 mat
        change = inversewmat[:, :-1]
        translation = inversewmat[:, -1].reshape((2, 1))
        for point in self.sift_points:
            coord = np.array(point.pt).reshape((2, 1))
            coord = np.matmul(change, coord)
            coord += translation
            coord.astype(np.int32)
            point.pt = (coord[0, 0], coord[1, 0])


def affine_detect_compute(img, phi, tilt, corners=None):
    # return SingleViewpointAsiftFeatures insstance
    detector = cv.xfeatures2d_SIFT.create()
    keypoint = SingleViewpointAsiftFeatures(tilt, phi)
    rot_img, rot_mat, rot_corners = rotation_phi(img, phi, corners)
    tilt_img, tilt_mat, tilt_corners = tilt_image(rot_img, tilt, rot_corners)
    keypoint.sift_points, keypoint.sift_desc = \
        detector.detectAndCompute(tilt_img, mask_of(tilt_img.shape[0], tilt_img.shape[1], tilt_corners))
    keypoint.normalize(rot_mat, tilt_mat)
    return keypoint


def detect_compute(img, content_corners=None, sigma_t=2 ** 0.5, n=5, b=72, draw=None):
    # asift keypoints
    if content_corners is None:
        content_corners = default_corners(img)
    detector = cv.xfeatures2d_SIFT.create()
    features = []
    first = SingleViewpointAsiftFeatures(1, 0)
    first.sift_points, first.sift_desc = detector. \
        detectAndCompute(img, mask_of(img.shape[0], img.shape[1], content_corners))
    features.append(first)
    for i in range(1, n + 1):
        tilt = sigma_t ** i
        k = 0
        phi = k * b / tilt
        while phi < 180:
            keypoint = affine_detect_compute(img, phi, tilt, content_corners)
            features.append(keypoint)
            # update phi
            k += 1
            phi = k * b / tilt
    if draw is not None:
        all_keypoints = []
        for asift_point in features:
            all_keypoints += asift_point.sift_points
        empty = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv.drawKeypoints(img, all_keypoints, empty)
        cv.imwrite(draw, empty)
    print("Detect complete !")
    return features


def two_resolution_match(img1, img2, corners1=None, corners2=None, draw=None, scale=1 / 3, n_affine=5,
                         match_result=None):
    ratio_thresh = 0.7
    if corners1 is None:
        corners1 = default_corners(img1)
    if corners2 is None:
        corners2 = default_corners(img2)
    corners1_ds = np.int32(corners1 * scale)
    corners2_ds = np.int32(corners2 * scale)
    img1_ds = cv.resize(img1, (0, 0), fx=scale, fy=scale)
    img2_ds = cv.resize(img2, (0, 0), fx=scale, fy=scale)
    img1_ds_features = detect_compute(img1_ds, corners1_ds)
    img2_ds_features = detect_compute(img2_ds, corners2_ds)
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    affine_pairs = []
    for ds_feature1 in img1_ds_features:
        for ds_feature2 in img2_ds_features:
            raw_matches = matcher.knnMatch(ds_feature1.sift_desc, ds_feature2.sift_desc, 2)
            good_matches = []
            for m, n in raw_matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            affine_pairs.append({'length': len(good_matches), 'affine1': (ds_feature1.phi, ds_feature1.tilt),
                                 'affine2': (ds_feature2.phi, ds_feature2.tilt)})
    good_affines = heapq.nlargest(n_affine, affine_pairs, key=lambda item: item['length'])
    all_matche_tupels = ImagePairAsiftMatches()
    print('Low resolution complete !')
    for affine in good_affines:
        points1, points2, matches = affine_feature_match(img1, img2, affine['affine1'], affine['affine2'], corners1,
                                                         corners2)
        all_matche_tupels.add_match(points1, points2, matches)
    img1_left_points, img2_left_points, all_matches = all_matche_tupels.getAll()
    match_filtering(img1_left_points, img2_left_points, all_matches)
    if draw is not None:
        draw_match(img1, img1_left_points, img2, img2_left_points, all_matches, draw)
    if match_result is not None:
        match_result.append((img1_left_points, img2_left_points, all_matches))
    corresponding1 = []
    corresponding2 = []
    for match in all_matches:
        corresponding1.append(img1_left_points[match.queryIdx].pt)
        corresponding2.append(img2_left_points[match.trainIdx].pt)
    corresponding1 = np.float32(corresponding1).reshape(-1, 1, 2)
    corresponding2 = np.float32(corresponding2).reshape(-1, 1, 2)
    return corresponding1, corresponding2


def affine_feature_match(img1, img2, affine1, affine2, corners1=None, corners2=None):
    ratio_thresh = 0.7
    if corners1 is None:
        corners1 = default_corners(img1)
    if corners2 is None:
        corners2 = default_corners(img2)
    img1feature = affine_detect_compute(img1, affine1[0], affine1[1], corners1)
    img2feature = affine_detect_compute(img2, affine2[0], affine2[1], corners2)
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    raw_matches = matcher.knnMatch(img1feature.sift_desc, img2feature.sift_desc, 2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return img1feature.sift_points, img2feature.sift_points, good_matches


def feature_match(img1, img2, corners1=None, corners2=None, img1_features=None, img2_features=None, draw=None):
    ratio_thresh = 0.7
    if img1_features is None:
        img1_features = detect_compute(img1, corners1)
    if img2_features is None:
        img2_features = detect_compute(img2, corners2)
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    all_matche_tupels = ImagePairAsiftMatches()
    turn = 0
    for img1feature in img1_features:
        for img2feature in img2_features:
            raw_matches = matcher.knnMatch(img1feature.sift_desc, img2feature.sift_desc, 2)
            turn += 1
            if turn % 20 == 0:
                print('asift match: ' + str(turn))
            good_matches = []
            # ratio test
            for m, n in raw_matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            all_matche_tupels.add_match(img1feature.sift_points, img2feature.sift_points, good_matches)
    img1_left_points, img2_left_points, all_matches = all_matche_tupels.getAll()
    match_filtering(img1_left_points, img2_left_points, all_matches)
    if draw is not None:
        draw_match(img1, img1_left_points, img2, img2_left_points, all_matches, draw)
    corresponding1 = []
    corresponding2 = []
    for match in all_matches:
        corresponding1.append(img1_left_points[match.queryIdx])
        corresponding2.append(img2_left_points[match.trainIdx])
    return corresponding1, corresponding2


def match_filtering(kpts1, kpts2, matches):
    print("left matches: " + str(len(matches)))
    distance = lambda pt1, pt2: abs(pt1.pt[0] - pt2.pt[0]) + abs(pt1.pt[1] - pt2.pt[1])
    isMultiple = lambda pt1, pt2: pt1.pt[0] - pt2.pt[0] >= 2 or pt1.pt[1] - pt2.pt[1] >= 2
    # remove identical matches
    idx1 = 0
    while idx1 < len(matches):
        if idx1 % 100 == 0:
            print('identical filtering: ' + str(idx1))
        remove_idx = []
        end1point1 = kpts1[matches[idx1].queryIdx]
        end2point1 = kpts2[matches[idx1].trainIdx]
        for idx2 in range(idx1 + 1, len(matches)):
            end1point2 = kpts1[matches[idx2].queryIdx]
            end2point2 = kpts2[matches[idx2].trainIdx]
            if distance(end1point1, end1point2) < 2 and distance(end2point1, end2point2) < 2:
                remove_idx.append(idx2)
        pop_num = 0
        for idx in remove_idx:
            matches.pop(idx - pop_num)
            pop_num += 1
        idx1 += 1
    print("left matches: " + str(len(matches)))
    # remove one to multiple matches
    idx1 = 0
    while idx1 < len(matches):
        if idx1 % 100 == 0:
            print('multiple filtering: ' + str(idx1))
        remove_first = False
        remove_idx = []
        end1point1 = kpts1[matches[idx1].queryIdx]
        end2point1 = kpts2[matches[idx1].trainIdx]
        for idx2 in range(idx1 + 1, len(matches)):
            end1point2 = kpts1[matches[idx2].queryIdx]
            end2point2 = kpts2[matches[idx2].trainIdx]
            if distance(end1point1, end1point2) < 2 and isMultiple(end2point1, end2point2):
                remove_first = True
                remove_idx.append(idx2)
        if remove_first:
            matches.pop(idx1)
            pop_num = 1
        else:
            idx1 += 1
            pop_num = 0
        for idx in remove_idx:
            matches.pop(idx - pop_num)
            pop_num += 1
    print("left matches: " + str(len(matches)))


if __name__ == '__main__':
    import os

    # test pair
    location = os.path.join(data_dir, 'Image', 'Cross-19-2019-11.png')
    location = cv.imread(location)
    map_img = os.path.join(data_dir, 'Image', 'BUAA-19-2019-11.png')
    map_img = cv.imread(map_img)
    augmented, corners = data_augment(location, expr_base)
    two_resolution_match(augmented, map_img, corners1=corners, draw=os.path.join(expr_base, 'fastmatch.png'))
    feature_match(augmented, map_img, corners1=corners, draw=os.path.join(expr_base, 'match.png'))
