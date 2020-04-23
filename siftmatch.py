from common import *


def detect_compute(img, content_corners=None, draw=None):
    if content_corners is None:
        content_corners = default_corners(img)
    detector = cv.xfeatures2d_SIFT.create()
    sift_points, sift_desc = detector.detectAndCompute(img, mask_of(img.shape[0], img.shape[1], content_corners))
    if draw is not None:
        empty = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv.drawKeypoints(img, sift_points, empty)
        cv.imwrite(draw, empty)
    print("Detect complete !")
    return sift_points, sift_desc


def feature_match(img1, img2, corners1=None, corners2=None, img1_features=None, img2_features=None, draw=None,
                  match_result=None):
    ratio_thresh = 0.7
    if img1_features is None:
        img1_points, img1_desc = detect_compute(img1, corners1)
    else:
        img1_points, img1_desc = img1_features[0], img1_features[1]
    if img2_features is None:
        img2_points, img2_desc = detect_compute(img2, corners2)
    else:
        img2_points, img2_desc = img2_features[0], img2_features[1]
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    raw_matches = matcher.knnMatch(img1_desc, img2_desc, 2)
    good_matches = []
    # match filtering
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    if draw is not None:
        draw_match(img1, img1_points, img2, img2_points, good_matches, draw)
    if match_result is not None:
        match_result.append((img1_points, img2_points, good_matches))
    corresponding1 = []
    corresponding2 = []
    for match in good_matches:
        corresponding1.append(img1_points[match.queryIdx].pt)
        corresponding2.append(img2_points[match.trainIdx].pt)
    corresponding1 = np.float32(corresponding1).reshape(-1, 1, 2)
    corresponding2 = np.float32(corresponding2).reshape(-1, 1, 2)
    return corresponding1, corresponding2


if __name__ == '__main__':
    import os

    # test pair
    map_path = os.path.join(data_dir, 'Image', 'map_village_scale_crop.jpg')
    frame_dir = os.path.join(data_dir, 'Image', 'frames', 'ds','rot')
    # frame_dir = os.path.join(data_dir, 'Image')
    frame = cv.imread(os.path.join(frame_dir, 'IMG_1126_ds_rot0.JPG'))
    reference = cv.imread(map_path)
    augmented_frame, corners = data_augment(frame, expr_base, scale_factor=0.6)
    match_result = []
    img1_points, img2_points = feature_match(augmented_frame, reference, corners1=corners,
                                             draw=os.path.join(expr_base, 'sift-match.png'),
                                             match_result=match_result)
    retval, mask = homography(augmented_frame, reference, img1_points, img2_points, src_corners=corners,
                              save_path=os.path.join(expr_base, 'sift-homography.png'))
    if retval is None:
        print('augment sift match failed !')
    else:
        valid_matches = []
        for i in range(mask.shape[0]):
            if mask[i][0] == 1:
                valid_matches.append(match_result[0][2][i])
        draw_match(augmented_frame, match_result[0][0], reference, match_result[0][1], valid_matches,
                   os.path.join(expr_base, 'corrected_sift_match.png'))
