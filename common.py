from cv2 import cv2 as cv
import numpy as np
import os

cwd = os.getcwd()
# proj_dir = os.path.dirname(cwd)
proj_dir = cwd
common_dir = os.path.dirname(proj_dir)

data_dir = os.path.join(common_dir, 'Datasets', 'UAVgeolocalization')
expr_base = os.path.join(proj_dir, 'experiments', 'match_task')


def warp_pts(pts_array, homo_mat):
    homog_pts = np.concatenate([pts_array, np.ones((pts_array.shape[0], 1))], axis=-1)
    warp_homog_pts = np.matmul(homog_pts, homo_mat.T)
    warp_homog_pts /= warp_homog_pts[:, 2:]
    return warp_homog_pts[:, :-1]


def unify_sift_desc(descriptors):
    current_length = (np.sum(descriptors ** 2, axis=-1)) ** 0.5 + 2e-8
    descs = descriptors / np.expand_dims(current_length, axis=1)
    descs[descs > 0.2] = 0.2
    thred_length = (np.sum(descs ** 2, axis=-1)) ** 0.5
    descs = descs / np.expand_dims(thred_length, axis=1)
    return descs


def cv_point(pt, orientation=0):
    point = cv.KeyPoint()
    point.size = 17
    point.angle = orientation
    point.class_id = -1
    point.octave = 0
    point.response = 0
    point.pt = (pt[0], pt[1])
    return point


def point_val(cvpoint):
    return cvpoint.pt, cvpoint.angle


def cv_match(queryIdx, trainIdx, distance=0):
    match = cv.DMatch()
    match.distance = distance
    match.queryIdx = queryIdx
    match.trainIdx = trainIdx
    match.imgIdx = 0
    return match


def neighbors(point, x_limit, y_limit, width=3, height=3, return_cvpt=False):
    assert width % 2 == 1 and height % 2 == 1
    neighbor_list = []
    x = int(point[0])
    y = int(point[1])
    for i in range(max(x - width // 2, x_limit[0]), min(x + width // 2 + 1, x_limit[1])):
        for j in range(max(y - height // 2, y_limit[0]), min(y + height // 2 + 1, y_limit[1])):
            if i != point[0] or j != point[1]:
                if not return_cvpt:
                    neighbor_list.append((i, j))
                else:
                    neighbor_list.append(cv_point((i, j)))
    return neighbor_list


def default_corners(img):
    return np.array([[0, 0], [img.shape[1] - 1, 0], [
        0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]])


def adaptive_affine(img, affine_mat, content_corners=None):
    # 2 by 2 mat
    # auto translation
    if content_corners is None:
        content_corners = default_corners(img)
    affined_corners = np.int32(np.matmul(content_corners, affine_mat.T))
    x, y, w, h = cv.boundingRect(affined_corners)
    translation = np.array([-x, -y])
    for corner in affined_corners:
        corner += translation
    # return affined and translated corners,adaptive translation affine mat,bounding rectangular width and height
    return affined_corners, np.concatenate([affine_mat, translation.reshape((2, 1))], axis=1), (w, h)


def affine_corners(corners, affine_mat):
    # 2 by 3 mat
    change = affine_mat[:, :-1]
    translation = np.array([affine_mat[0, -1], affine_mat[1, -1]])
    changed = np.matmul(corners, change.T)
    for corner in changed:
        corner += translation
    return np.int32(changed)


def data_augment(img, save, scale_factor=1, light=True):
    # random affine transform--keep scale,make translation and rotation,change viewpoint
    if scale_factor != 1:
        scaled_img = cv.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    else:
        scaled_img = img
    if light:
        max_tilt = 3
        max_rotation = 60
        max_spin = 60
    else:
        max_tilt = 6
        max_rotation = 360
        max_spin = 360
    filename = os.path.join(save, 'augmented.png')
    # cropname = os.path.join(save, 'crop.png')
    rand_tilt = np.random.randint(1, max_tilt)
    rand_img_rotation = np.random.random() * max_rotation
    rand_camera_spin = np.random.random() * max_spin
    rot_img, rot_mat, rot_corners = rotation_phi(scaled_img, rand_img_rotation)
    spin_img, spin_mat, spin_corners = rotation_phi(rot_img, rand_camera_spin, content_corners=rot_corners)
    result, tilt_mat, tilt_corners = tilt_image(spin_img, rand_tilt, spin_corners)
    cv.imwrite(filename, result)
    constant = np.array([[0, 0, 1]])
    scale_mat = np.eye(3)
    scale_mat[0][0] = scale_factor
    scale_mat[1][1] = scale_factor
    whole_mat = np.matmul(np.concatenate([rot_mat, constant]), scale_mat)
    whole_mat = np.matmul(np.concatenate([spin_mat, constant]), whole_mat)
    whole_mat = np.matmul(np.concatenate([tilt_mat, constant]), whole_mat)
    print(np.linalg.inv(whole_mat))
    return result, tilt_corners


def rotation_phi(img, phi, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    if phi == 0:
        return img, np.concatenate([np.eye(2), np.zeros((2, 1))], axis=1), content_corners
    phi = np.deg2rad(phi)
    s, c = np.sin(phi), np.cos(phi)
    mat_rot = np.array([[c, -s], [s, c]])
    rot_corners, affine_mat, bounding = adaptive_affine(img, mat_rot, content_corners)
    affined = cv.warpAffine(img, affine_mat, bounding)
    return affined, affine_mat, rot_corners


def adaptive_scale(img, factor, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    mat_scale = np.array([[factor, 0], [0, factor]])
    scale_corners, affine_mat, bounding = adaptive_affine(img, mat_scale, content_corners)
    scaled = cv.warpAffine(img, affine_mat, bounding)
    return scaled, affine_mat, scale_corners


def tilt_image(img, tilt, content_corners=None):
    if content_corners is None:
        content_corners = default_corners(img)
    if tilt == 1:
        return img, np.concatenate([np.eye(2), np.zeros((2, 1))], axis=1), content_corners
    gaussian_sigma = 0.8 * np.sqrt(tilt ** 2 - 1)
    unti_aliasing = cv.GaussianBlur(
        img, (3, 1), sigmaX=0, sigmaY=gaussian_sigma)
    mat_tilt = np.array([[1, 0], [0, 1 / tilt]])
    tilt_corners, affine_mat, bounding = adaptive_affine(img, mat_tilt, content_corners)
    affined = cv.warpAffine(unti_aliasing, affine_mat, bounding)
    return affined, affine_mat, tilt_corners


def mask_of(h, w, corners):
    mask = np.zeros((h, w), np.uint8)
    sequence_corners = np.array(
        [corners[0], corners[1], corners[3], corners[2]]).reshape((-1, 1, 2))
    cv.polylines(mask, [sequence_corners], True, 255)
    cv.fillPoly(mask, [sequence_corners], 255)
    return mask


def draw_match(img1, img1_points, img2, img2_points, matches, path, group_draw=None, group_draw_preffix=None,
               homo_mat=None):
    if group_draw is None:
        result = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(img1, img1_points, img2, img2_points, matches, result,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(path, result)
    else:
        # assert type(group_draw_preffix) == 'str'
        # assert len(crop_img2) == 4
        if not os.path.exists(path):
            os.mkdir(path)
        indixes = np.array(list(range(len(matches))))
        np.random.shuffle(indixes)
        group_start = 0
        draw_index = 0
        while group_start + group_draw < len(indixes):
            try:
                draw_matches = []
                pts1 = []
                pts2 = []
                for i in range(group_start, group_start + group_draw):
                    draw_match = matches[indixes[i]]
                    new_match = cv_match(i - group_start, i - group_start, draw_match.distance)
                    draw_matches.append(new_match)
                    pts1.append(cv_point(img1_points[draw_match.queryIdx].pt))
                    pts2.append(cv_point(img2_points[draw_match.trainIdx].pt))

                pts1_array = np.array([point.pt for point in pts1])
                x_min, x_max, y_min, y_max = pts1_array[:, 0].min(), pts1_array[:, 0].max(), \
                                             pts1_array[:, 1].min(), pts1_array[:, 1].max()
                x_min, x_max = max(0, x_min - 128), min(img1.shape[1], x_max + 128)
                y_min, y_max = max(0, y_min - 128), min(img1.shape[0], y_max + 128)
                region = np.array([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
                warp_region = warp_pts(region, homo_mat)
                crop2 = np.int32((warp_region[:, 0].min(), warp_region[:, 0].max(),
                                  warp_region[:, 1].min(), warp_region[:, 1].max()))
                crop_img1 = img1[int(y_min):int(y_max), int(x_min):int(x_max), :].copy()
                refx_min, refx_max = max(0, crop2[0] - 128), min(img2.shape[1], crop2[1] + 128)
                refy_min, refy_max = max(0, crop2[2] - 128), min(img2.shape[0], crop2[3] + 128)
                ref_img = img2[refy_min:refy_max, refx_min:refx_max, :].copy()
                for i in range(group_draw):
                    coord1 = pts1[i].pt
                    pts1[i].pt = (coord1[0] - x_min, coord1[1] - y_min)
                    coord2 = pts2[i].pt
                    pts2[i].pt = (coord2[0] - refx_min, coord2[1] - refy_min)
                    '''draw_match = draw_matches[i]
                    pt2_coord = pts2[i].pt
                    cv.putText(ref_img, str(draw_match.distance), (int(pt2_coord[0] + 10), int(pt2_coord[1])), 0,
                               5e-3 * 200, (255, 0, 0))'''
                result = np.empty((max(crop_img1.shape[0], ref_img.shape[0]), crop_img1.shape[1] + ref_img.shape[1], 3),
                                  dtype=np.uint8)
                cv.drawMatches(crop_img1, pts1, ref_img, pts2, draw_matches, result,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imwrite(os.path.join(path, group_draw_preffix + str(draw_index) + '.jpg'), result)
            except BaseException as e:
                print(e)
            group_start += group_draw
            draw_index += 1


def homography(src_img, dst_img, src_points, dst_points, save_path=None, src_corners=None, ransac_thrd=1,
               ransac_iter=4096):
    if src_corners is None:
        src_corners = default_corners(src_img)
    retval, mask = cv.findHomography(src_points, dst_points, method=cv.RANSAC, ransacReprojThreshold=ransac_thrd,
                                     maxIters=ransac_iter)
    if save_path is not None:
        try:
            homogeneous_corners = np.concatenate([src_corners, np.ones((4, 1))], axis=1)
            projected_corners = np.matmul(homogeneous_corners, retval.T)
            for coordinate in projected_corners:
                coordinate /= coordinate[-1]
            projected_corners = np.int32(projected_corners[:, :-1])
            # x, y, w, h = cv.boundingRect(projected_corners)
            perspective = cv.warpPerspective(src_img, retval, (dst_img.shape[1], dst_img.shape[0]))
            background = cv.cvtColor(dst_img, cv.COLOR_BGR2GRAY)
            background = cv.cvtColor(background, cv.COLOR_GRAY2BGR)
            patched_img = patch2background(perspective, projected_corners, background)
            cv.imwrite(save_path, patched_img)
        except BaseException as e:
            print(e)
    return retval, mask


def patch2background(src_img, content_corner, background):
    poly_corners = np.array([content_corner[0], content_corner[1], content_corner[3], content_corner[2]])
    mask = np.zeros((src_img.shape[0], src_img.shape[1]))
    mask = cv.fillPoly(mask, [poly_corners], color=1)
    min_x = poly_corners[:, 0].min()
    max_x = poly_corners[:, 0].max()
    min_y = poly_corners[:, 1].min()
    max_y = poly_corners[:, 1].max()
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if mask[y][x] == 1:
                background[y][x] = src_img[y][x]
    return background


def center_of_corners(corners):
    k1 = (corners[1][1] - corners[0][1]) / (corners[1][0] - corners[0][0])
    b1 = (corners[1][0] * corners[0][1] - corners[0][0] * corners[1][1]) / (corners[1][0] - corners[0][0])
    k2 = (corners[3][1] - corners[2][1]) / (corners[3][0] - corners[2][0])
    b2 = (corners[3][0] * corners[2][1] - corners[2][0] * corners[3][1]) / (corners[3][0] - corners[2][0])
    x = (b2 - b1) / (k1 - k2)
    y = k1 * (b2 - b1) / (k1 - k2) + b1
    return np.array((x, y))


def warp_error(test_pts, target_pts, homo_mat, log=None):
    test_points_array = np.array(test_pts)
    homog_tests = np.concatenate([test_points_array, np.ones((len(test_pts), 1))], axis=1)
    homog_estimates = np.matmul(homog_tests, homo_mat.T)
    homog_estimates /= homog_estimates[:, 2:]
    target_array = np.array(target_pts)
    estimate_array = homog_estimates[:, :-1]
    square_error = (target_array - estimate_array) ** 2
    point_wise_error = (np.sum(square_error, axis=-1)) ** 0.5
    average_error = np.mean(point_wise_error, axis=-1)
    if log is not None:
        log.info('Test points: ')
        log.info(str(test_points_array))
        log.info('Target points: ')
        log.info(str(target_array))
        log.info('Estimated points: ')
        log.info(str(estimate_array))
        log.info('Point-wise error: ')
        log.info(str(point_wise_error))
        log.info('Average error: ')
        log.info(str(average_error))


if __name__ == '__main__':
    '''origin_frame_dir = os.path.join(data_dir, 'Image', 'Village0', 'original frames')
    out_dir = os.path.join(data_dir, 'Image', 'Village0', 'anno_corrected_loc')
    files = os.listdir(origin_frame_dir)
    anno_corners = []
    for filename in files:
        img = cv.imread(os.path.join(origin_frame_dir, filename))
        scaled, scale_mat, scale_corners = adaptive_scale(img, 0.15)
        roted, rot_mat, rot_corners = rotation_phi(scaled, -132, scale_corners)
        cv.imwrite(os.path.join(out_dir, filename), roted)
        anno_corners.append(rot_corners)
    print(anno_corners)'''
    '''img = patch2background()
    cv.imwrite('./patch.jpg', img)'''
    import pandas as pd

    pt_file = '/home/patrick/PatrickWorkspace/Datasets/UAVgeolocalization/Image/Village0/' \
              'loc/loc2_centered_correspondences.csv'
    pt_file = '/home/patrick/PatrickWorkspace/Datasets/UAVgeolocalization/Image/Village0/' \
              'crop_loc/balance_crops/loc2_3_centered_correspondences.csv'
    pt_csv = pd.read_csv(pt_file)
    pt_indices = pt_csv.index
    test_pts = []
    target_pts = []
    for idx in pt_indices:
        loc_corners = np.array([[pt_csv.loc[idx, 'loc_x1'], pt_csv.loc[idx, 'loc_y1']],
                                [pt_csv.loc[idx, 'loc_x2'], pt_csv.loc[idx, 'loc_y2']],
                                [pt_csv.loc[idx, 'loc_x3'], pt_csv.loc[idx, 'loc_y3']],
                                [pt_csv.loc[idx, 'loc_x4'], pt_csv.loc[idx, 'loc_y4']]])
        test_pts.append(center_of_corners(loc_corners))
        map_corners = np.array([[pt_csv.loc[idx, 'map_x1'], pt_csv.loc[idx, 'map_y1']],
                                [pt_csv.loc[idx, 'map_x2'], pt_csv.loc[idx, 'map_y2']],
                                [pt_csv.loc[idx, 'map_x3'], pt_csv.loc[idx, 'map_y3']],
                                [pt_csv.loc[idx, 'map_x4'], pt_csv.loc[idx, 'map_y4']]])
        target_pts.append(center_of_corners(map_corners))
    # loc neighbor refine
    retval = np.array([[9.46871526e-01, -1.01905502e+00, 3.85965237e+03],
                       [1.00280253e-01, 3.68942000e-01, 1.20087459e+03],
                       [4.92943751e-05, -2.34958096e-04, 1.00000000e+00]])
    # ncc refine
    retval = np.array([[6.72028576e-01, -7.21392580e-01, 3.85789143e+03],
                       [-2.32925212e-02, 5.12697781e-01, 1.19038493e+03],
                       [-1.57932502e-05, -1.65508968e-04, 1.00000000e+00]])
    # loc neighbor unit
    retval = np.array([[6.87231054e-01, -9.95046332e-01, 3.86841524e+03],
                       [-4.42473396e-03, 3.67276128e-01, 1.21040320e+03],
                       [-9.42593135e-06, -2.27342257e-04, 1.00000000e+00]])
    # map neighbor refine
    retval = np.array([[6.29182113e-01, -8.06677224e-01, 3.86403436e+03],
                       [-3.63584785e-02, 4.51432594e-01, 1.20949732e+03],
                       [-2.46397052e-05, - 1.84269618e-04, 1.00000000e+00]])
    # loc2_1
    retval = np.array([[1.00155061e+00, -1.20699765e+00, 4.27479258e+03],
                       [8.02890018e-02, 4.09408704e-01, 1.67747176e+03],
                       [3.81050255e-05, -2.76158193e-04, 1.00000000e+00]])
    # loc2_2
    retval = np.array([[-1.36908449e+01, -9.03569888e-01, 4.06420903e+03],
                       [-5.45102474e+00, -1.43769223e-01, 1.56609601e+03],
                       [-3.38855602e-03, -2.18137033e-04, 1.00000000e+00]])
    # loc2_3
    retval = np.array([[-2.33242695e+00, 1.07429069e+00, 4.20984550e+03],
                       [-1.19278636e+00, 1.09960441e+00, 1.49175137e+03],
                       [-7.00629802e-04, 2.51175618e-04, 1.00000000e+00]])
    warp_error(test_pts, target_pts, retval)
    '''origin = cv.imread('/home/patrick/PatrickWorkspace/AerialVisualGeolocalization/experiments/origin.png')
    trans = cv.imread('/home/patrick/PatrickWorkspace/AerialVisualGeolocalization/experiments/trans.png')
    detector = cv.xfeatures2d_SIFT.create()
    pts1, descs1 = detector.detectAndCompute(origin, None)
    pts2, descs2 = detector.detectAndCompute(trans, None)
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    ratio_thresh = 0.75
    raw_matches = matcher.knnMatch(descs1, descs2, 2)
    good_matches = []
    # match filtering
    match_cvpt1 = []
    match_cvpt2 = []
    match_pts1 = []
    match_pts2 = []
    count = 0
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            # heapq.heappush(good_matches, (m.distance, m))
            match_cvpt1.append(pts1[m.queryIdx])
            match_pts1.append(pts1[m.queryIdx].pt)
            match_cvpt2.append(pts2[m.trainIdx])
            match_pts2.append(pts2[m.trainIdx].pt)
            good_matches.append(cv_match(count, count))
            count += 1
    crop = (0, trans.shape[1], 0, trans.shape[0])
    match_pts1 = np.array(match_pts1).reshape((-1, 1, 2))
    match_pts2 = np.array(match_pts2).reshape((-1, 1, 2))
    retval, mask = homography(origin, trans, match_pts1, match_pts2)

    draw_match(origin, match_cvpt1, trans, match_cvpt2, good_matches,
               '/home/patrick/PatrickWorkspace/AerialVisualGeolocalization/experiments/draw_test',
               8, 'draw', retval)
    draw_match(origin, match_cvpt1, trans, match_cvpt2, good_matches,
               '/home/patrick/PatrickWorkspace/AerialVisualGeolocalization/experiments/draw_test/corrected_slic_match.png')
    '''
