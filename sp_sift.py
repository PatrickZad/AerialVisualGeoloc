import cv2 as cv
import pickle
import common
import numpy as np
import pandas as pd
from common import data_dir
import os


def ncc_extrema(t, area):
    response = cv.matchTemplate(
        np.uint8(area), np.uint8(t), method=cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(response)
    return max_loc


def subpx_ncc_extrema(t, area):
    # 0 padding for edge response
    max_iter = 5
    response = cv.matchTemplate(
        np.uint8(area), np.uint8(t), method=cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(response)
    pad_response = np.zeros(area.shape[:2]) + 0.0
    offset = ((area.shape[1] - response.shape[1]) // 2,
              (area.shape[0] - response.shape[0]) // 2)
    pad_response[offset[1]:offset[1] + response.shape[0],
                 offset[0]:offset[0] + response.shape[1]] = response
    iter = 0
    max_loc = [max_loc[0], max_loc[1]]
    x, y = max_loc
    subpx_max_loc = (x, y)
    while iter < max_iter:
        next_iter = False
        dx = (pad_response[y][x + 1] - pad_response[y][x - 1]) / 2
        dy = (pad_response[y + 1][x] - pad_response[y - 1][x]) / 2
        dxx = pad_response[y][x + 1] - 2 * \
            pad_response[y][x] + pad_response[y][x - 1]
        dxy = ((pad_response[y + 1][x + 1] - pad_response[y + 1][x - 1]) - (
            pad_response[y - 1][x + 1] - pad_response[y - 1][x - 1])) / 4.
        dyy = pad_response[y + 1][x] - 2 * \
            pad_response[y][x] + pad_response[y - 1][x]
        Jac = np.array([dx, dy])
        Hes = np.array([[dxx, dxy], [dxy, dyy]])
        try:
            coord_off = -np.linalg.inv(Hes).dot(Jac)
            if coord_off[0] > 0.5:
                max_loc[0] += 1
                next_iter = True
            if coord_off[1] > 0.5:
                max_loc[1] += 1
                next_iter = True
            if next_iter:
                iter += 1
                continue
            subpx_max_loc = (max_loc[0] + coord_off[0],
                             max_loc[1] + coord_off[1])
            break
        except Exception as e:
            break
    return subpx_max_loc


detect_methods = ('slic', 'seeds')
# subpx use maximum estimation to obtain subpixel-level match
refine_methods = {'ncc': ncc_extrema, 'sbpx_ncc': subpx_ncc_extrema}


def cv_point(pt, orientation=0):
    point = cv.KeyPoint()
    point.size = 17
    point.angle = orientation
    point.class_id = -1
    point.octave = 0
    point.response = 0
    point.pt = (pt[0], pt[1])
    return point


class simple_point:
    def __init__(self, cv_point=None):
        if cv_point is None:
            self.size = 17
            self.angle = 0
            self.class_id = -1
            self.octave = 0
            self.response = 0
            self.pt = (0.0, 0.0)
        else:
            self.size = cv_point.size
            self.angle = cv_point.angle
            self.class_id = cv_point.class_id
            self.octave = cv_point.octave
            self.response = cv_point.response
            self.pt = cv_point.pt

    def cvt2cvpt(self):
        point = cv.KeyPoint()
        point.size = self.size
        point.angle = self.angle
        point.class_id = self.class_id
        point.octave = self.octave
        point.response = self.response
        point.pt = self.pt
        return point


def default_corners(img):
    return np.array([[0, 0], [img.shape[1] - 1, 0],
                     [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]])


def mask_of(h, w, content_corners):
    mask = np.zeros((h, w), np.uint8)
    sequence_corners = content_corners.reshape((-1, 1, 2))
    cv.polylines(mask, [sequence_corners], True, 255)
    cv.fillPoly(mask, [sequence_corners], 255)
    return mask


def keypoint_detect(img, algorithm, content_corners=None, cell_size=16, save=None):
    assert algorithm in detect_methods
    if content_corners is None:
        content_corners = default_corners(img)
    if algorithm == detect_methods[0]:
        lab_img = common.cvt2lab(img)
        detector = cv.ximgproc.createSuperpixelSLIC(
            lab_img, region_size=cell_size)
        detector.iterate()
    elif algorithm == detect_methods[1]:
        h, w = img.shape[:2]
        cell_num = h * w // (cell_size * 2**5)
        c = img.shape[-1] if len(img.shape) > 2 else 1
        detector = cv.ximgproc.createSuperpixelSEEDS(
            w, h, c, cell_num, 4)
        detector.iterate(img)
    spx_mask = detector.getLabelContourMask()
    content_mask = mask_of(img.shape[0], img.shape[1], content_corners)
    pt_mask = spx_mask == content_mask
    x_coords = np.repeat(np.arange(img.shape[1]).reshape(
        (1, -1)), img.shape[0], axis=0)
    y_coords = np.repeat(np.arange(img.shape[0]).reshape(
        (-1, 1)), img.shape[1], axis=-1)
    pt_coords = np.concatenate(
        [np.expand_dims(x_coords, axis=-1), np.expand_dims(y_coords, axis=-1)], axis=-1)
    keypt_coords = pt_coords[pt_mask]
    result = []
    for i in range(keypt_coords.shape[0]):
        result.append(cv_point(keypt_coords[i, :]))
    if save is not None:
        save_img = img.copy()
        save_img[pt_mask] = (0, 0, 255)
        cv.imwrite(save, save_img)
    return result


def compute_desc(img=None, keypoints=None, save=None, load=None):
    if load is not None:
        with open(load, 'rb') as file:
            feat_pair = pickle.load(file)
        pts, descriptors = feat_pair
        points = [pt.cvt2cvpt() for pt in pts]
    else:
        detector = cv.xfeatures2d_SIFT.create()
        points, descriptors = detector.compute(img, keypoints)
        if save is not None:
            pts = [simple_point(cv_pt) for cv_pt in points]
            with open(save, 'wb') as file:
                pickle.dump((pts, descriptors), file)
    return points, descriptors


def img_match(query, target, ref_method, query_corners=None, target_corners=None, q_feats=None, t_feats=None, det_method=None, save=None):
    assert ref_method in refine_methods
    if q_feats is None:
        pts_q = keypoint_detect(query, det_method, query_corners)
        pts_q, descs_q = compute_desc(query, pts_q)
    else:
        pts_q, descs_q = q_feats
    if t_feats is None:
        pts_t = keypoint_detect(target, det_method, target_corners)
        pts_t, descs_t = compute_desc(query, pts_t)
    else:
        pts_t, descs_t = t_feats
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    raw_matches = matcher.knnMatch(descs_q, descs_t, 2)
    match_coords_query = []
    match_coords_target = []
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            query_pt = pts_q[m.queryIdx]
            target_pt = pts_t[m.trainIdx]
            match_coords_query.append(query_pt.pt)
            match_coords_target.append(target_pt.pt)
            m.trainIdx = len(match_coords_target) - 1
            m.queryIdx = len(match_coords_query) - 1
            good_matches.append(m)

    match_coords_target, match_cvpts_target, match_cvpts_query = ncc_refine(
        query, target, match_coords_query, match_coords_target, ref_method)
    query_pts = np.array(match_coords_query).reshape((-1, 1, 2))
    target_pts = np.array(match_coords_target).reshape((-1, 1, 2))
    homog, mask = cv.findHomography(
        query_pts, target_pts, method=cv.RANSAC, ransacReprojThreshold=1, maxIters=4096)
    if save is not None:
        filtered_matches = [good_matches[k]
                            for k in range(len(mask)) if mask[k] == 1]
        save_match(query, target, match_cvpts_query,
                   match_cvpts_target, filtered_matches, save)
    return homog, query_pts[mask == 1], target_pts[mask == 1]


def save_match(query, target, query_cvpts_all, target_cvpts_all, matches, file_path):
    result = np.empty((max(query.shape[0], target.shape[0]),
                       query.shape[1] + target.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(query, query_cvpts_all, target, target_cvpts_all,
                   matches, result, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(file_path, result)


def ncc_refine(img1, img2, img1_pts, img2_pts, method, t_w=15, s_w=17):
    # 15*15 template,17*17 search are
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    match_coords_target, match_cvpts_target, match_cvpts_query = [], [], []
    for i in range(len(img1_pts)):
        pt_q = img1_pts[i]
        pt_t = img2_pts[i]
        match_cvpts_query.append(cv_point(pt_q))
        max_t_expand = (t_w - 1) // 2
        min_edge_distance_q = min(
            pt_q[0], pt_q[1], img1.shape[1] - 1 - pt_q[0], img1.shape[0] - 1 - pt_q[1])
        t_expand = int(min(max_t_expand, min_edge_distance_q))
        if t_expand == 0 or pt_t[0] - t_expand <= 0 or pt_t[0] + t_expand >= img2.shape[1] \
                or pt_t[1] - t_expand <= 0 or pt_t[1] + t_expand >= img2.shape[0]:
            match_coords_target.append(pt_t)
            match_cvpts_target.append(cv_point(pt_t))
            continue
        max_area_expand = (s_w - 1) // 2
        min_edge_distance_t = min(
            pt_t[0], pt_t[1], img2.shape[1] - 1 - pt_t[0], img2.shape[0] - 1 - pt_t[1])
        s_expand = int(min(t_expand + max_area_expand, min_edge_distance_t))
        a_expand = s_expand - t_expand
        pt_q = (int(pt_q[0]), int(pt_q[1]))
        pt_t = (int(pt_t[0]), int(pt_t[1]))
        img_t = img1[pt_q[1] - t_expand:pt_q[1] + t_expand + 1,
                     pt_q[0] - t_expand:pt_q[0] + t_expand + 1, :]
        img_s = img2[pt_t[1] - s_expand:pt_t[1] + s_expand + 1,
                     pt_t[0] - s_expand:pt_t[0] + s_expand + 1, :]
        offset = refine_methods[method](img_t, img_s)
        refine_pt = (pt_t[0] - a_expand + offset[0],
                     pt_t[1] - a_expand + offset[1])
        match_coords_target.append(refine_pt)
        match_cvpts_target.append(cv_point(refine_pt))
    return match_coords_target, match_cvpts_target, match_cvpts_query


def center_of_corners(corners):
    k1 = (corners[1][1] - corners[0][1]) / (corners[1][0] - corners[0][0])
    b1 = (corners[1][0] * corners[0][1] - corners[0][0] *
          corners[1][1]) / (corners[1][0] - corners[0][0])
    k2 = (corners[3][1] - corners[2][1]) / (corners[3][0] - corners[2][0])
    b2 = (corners[3][0] * corners[2][1] - corners[2][0] *
          corners[3][1]) / (corners[3][0] - corners[2][0])
    x = (b2 - b1) / (k1 - k2)
    y = k1 * (b2 - b1) / (k1 - k2) + b1
    return np.array((x, y))


class VillageReader:
    def __init__(self):
        self.map_path = os.path.join(data_dir, 'Image', 'Village0', 'map.jpg')
        self.frame_dir = os.path.join(data_dir, 'Image', 'Village0', 'loc')
        base_dir = os.path.join(data_dir, 'Image', 'Village0')
        self.frame_dir = os.path.join(base_dir, 'loc')
        self.corners = pd.read_csv(os.path.join(self.frame_dir, 'corners.csv'))
        self._next_id = 1

    def map_img(self):
        return cv.imread(self.map_path)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_id > 15:
            raise StopIteration
        id = self._next_id
        fileid = 'loc' + str(id)
        frame_file = 'loc' + str(id) + '.JPG'
        frame = cv.imread(os.path.join(self.frame_dir, frame_file))
        corner = np.array([[self.corners.loc[frame_file, 'x1'], self.corners.loc[frame_file, 'y1']],
                           [self.corners.loc[frame_file, 'x2'],
                               self.corners.loc[frame_file, 'y2']],
                           [self.corners.loc[frame_file, 'x3'],
                               self.corners.loc[frame_file, 'y3']],
                           [self.corners.loc[frame_file, 'x0'], self.corners.loc[frame_file, 'y0']]])
        pt_file = os.path.join(self.frame_dir, 'loc' + str(id) + '.csv')
        pt_df = pd.read_csv(pt_file)
        test_array = np.array([
            center_of_corners(np.reshape(
                pt_df.loc['pt1', 'loc_x1':'loc_y4'].values, (4, 2))),
            center_of_corners(np.reshape(
                pt_df.loc['pt2', 'loc_x1':'loc_y4'].values, (4, 2))),
            center_of_corners(np.reshape(
                pt_df.loc['pt3', 'loc_x1':'loc_y4'].values, (4, 2))),
            center_of_corners(np.reshape(
                pt_df.loc['pt4', 'loc_x1':'loc_y4'].values, (4, 2)))
        ])
        target_array = np.array([
            center_of_corners(np.reshape(
                pt_df.loc['pt1', 'map_x1':].values, (4, 2))),
            center_of_corners(np.reshape(
                pt_df.loc['pt2', 'map_x1':].values, (4, 2))),
            center_of_corners(np.reshape(
                pt_df.loc['pt3', 'map_x1':].values, (4, 2))),
            center_of_corners(np.reshape(
                pt_df.loc['pt4', 'map_x1':].values, (4, 2))),
        ])
        self._next_id += 2
        return fileid, frame, corner, test_array, target_array


def warp_error(test_array, target_array, homo_mat):
    homo_coord_tests = np.concatenate([test_array, np.ones(
        (test_array.shape[0], 1))], axis=-1)
    homog_estimates = np.matmul(homo_coord_tests, homo_mat.T)
    homog_estimates /= homog_estimates[..., -1:]
    estimate_array = homog_estimates[..., :-1]
    square_error = (target_array - estimate_array) ** 2
    point_wise_error = (np.sum(square_error, axis=-1)) ** 0.5
    average_error = np.mean(point_wise_error, axis=-1)
    return {'point_wise_error': point_wise_error, 'avg_error': average_error}


if __name__ == '__main__':
    # target = cv.imread(map_path)
    expr_common = os.path.join('./experiments', 'comparison')
    # debug
    '''img_q = cv.imread(os.path.join(expr_common, 'test_query.jpg'))
    img_t = cv.imread(os.path.join(expr_common, 'test_target.jpg'))
    q_pt_slic = keypoint_detect(
        img_q, 'slic', save=os.path.join(expr_common, 'slic_query.jpg'))
    q_pt_seeds = keypoint_detect(
        img_q, 'seeds', save=os.path.join(expr_common, 'seeds_query.jpg'))
    t_pt_slic = keypoint_detect(img_t, 'slic')
    t_pt_seeds = keypoint_detect(img_t, 'seeds')
    q_pt_slic, q_desc_slic = compute_desc(img_q, q_pt_slic)
    q_pt_seeds, q_desc_seeds = compute_desc(img_q, q_pt_seeds)
    t_pt_slic, t_desc_slic = compute_desc(
        img_t, t_pt_slic, save=os.path.join(expr_common, 'debug_slic.pkl'))
    t_pt_seeds, t_desc_seeds = compute_desc(
        img_t, t_pt_seeds, save=os.path.join(expr_common, 'debug_seeds.pkl'))
    homog1, qpt1, tpt1 = img_match(img_q, img_t, 'ncc', q_feats=(q_pt_slic, q_desc_slic), t_feats=(
        t_pt_slic, t_desc_slic), save=os.path.join(expr_common, 'match_slic_ncc.jpg'))
    homog2, qpt2, tpt2 = img_match(img_q, img_t, 'sbpx_ncc', q_feats=(q_pt_slic, q_desc_slic), t_feats=(
        t_pt_slic, t_desc_slic), save=os.path.join(expr_common, 'match_slic_sub_ncc.jpg'))
    homog3, qpt3, tpt3 = img_match(img_q, img_t, 'ncc', q_feats=(q_pt_seeds, q_desc_seeds), t_feats=(
        t_pt_slic, t_desc_slic), save=os.path.join(expr_common, 'match_seeds_ncc.jpg'))
    homog4, qpt4, tpt4 = img_match(img_q, img_t, 'sbpx_ncc', q_feats=(q_pt_seeds, q_desc_seeds), t_feats=(
        t_pt_slic, t_desc_slic), save=os.path.join(expr_common, 'match_seeds_sub_ncc.jpg'))'''

    '''test'''
    img_reader = VillageReader()
    target_img = img_reader.map_img()
    # target_pt_slic = keypoint_detect(
    #    target_img, 'slic', save=os.path.join(expr_common, 'test_slic.jpg'))
    # target_feat_slic = compute_desc(
    #    target_img, target_pt_slic, save=os.path.join(expr_common, 'test_slic.pkl'))
    # target_feat_slic = compute_desc(
    #    load=os.path.join(expr_common, 'test_slic.pkl'))
    # target_pt_seeds = keypoint_detect(
    #    target_img, 'seeds', save=os.path.join(expr_common, 'test_seeds.jpg'))
    # target_feat_seeds = compute_desc(
    #    target_img, target_pt_seeds, save=os.path.join(expr_common, 'test_seeds.pkl'))
    target_feat_seeds = compute_desc(
        load=os.path.join(expr_common, 'test_seeds.pkl'))
    for img_id, query_img, corner, test_array, target_array in img_reader:
        print(img_id)

        '''
        query_pt_slic = keypoint_detect(query_img, 'slic')
        query_feat_slic = compute_desc(query_img, query_pt_slic)
        '''
        query_pt_seeds = keypoint_detect(query_img, 'seeds')
        query_feat_seeds = compute_desc(query_img, query_pt_seeds)
        '''
        # slic vs seeds, both use ncc refine
        
        homog_slic, qpt_slic, tpt_slic = img_match(query_img, target_img, 'ncc', corner, q_feats=query_feat_slic,
                                                   t_feats=target_feat_slic, save=os.path.join(expr_common, 'superpixel', 'slic_match_' + img_id + '.jpg'))
        
        homog_seeds, qpt_seeds, tpt_seeds = img_match(query_img, target_img, 'ncc', corner, q_feats=query_feat_seeds,
                                                      t_feats=target_feat_seeds, save=os.path.join(expr_common, 'superpixel', 'seeds_match_' + img_id + '.jpg'))
        
        slic_error = warp_error(test_array, target_array, homog_slic)
        
        seeds_error = warp_error(test_array, target_array, homog_seeds)
        
        slic_fit_error = warp_error(qpt_slic.reshape(
            (-1, 2)), tpt_slic.reshape((-1, 2)), homog_slic)
        
        seeds_fit_error = warp_error(qpt_seeds.reshape(
            (-1, 2)), tpt_seeds.reshape((-1, 2)), homog_seeds)
        
        print('slic_ncc')
        print(slic_error)
        print('pts: ' + str(qpt_slic.shape[0]))
        print(slic_fit_error)
        
        print('seeds_ncc')
        print(seeds_error)
        print('pts: ' + str(qpt_seeds.shape[0]))
        print(seeds_fit_error)
        
        '''
        # ncc vs subpx_ncc , both use seeds
        homog_ncc, qpt_ncc, tpt_ncc = img_match(query_img, target_img, 'ncc', corner, q_feats=query_feat_seeds,
                                                t_feats=target_feat_seeds, save=os.path.join(expr_common, 'subpixel_refine', 'ncc_match_' + img_id + '.jpg'))
        homog_sbpx, qpt_sbpx, tpt_sbpx = img_match(query_img, target_img, 'sbpx_ncc', corner, q_feats=query_feat_seeds,
                                                   t_feats=target_feat_seeds, save=os.path.join(expr_common, 'subpixel_refine', 'subpx_match_' + img_id + '.jpg'))
        ncc_error = warp_error(test_array, target_array, homog_ncc)
        sbpx_error = warp_error(test_array, target_array, homog_sbpx)
        ncc_fit_error = warp_error(qpt_ncc.reshape(
            (-1, 2)), tpt_ncc.reshape((-1, 2)), homog_ncc)
        sbpx_fit_error = warp_error(qpt_sbpx.reshape(
            (-1, 2)), tpt_sbpx.reshape((-1, 2)), homog_sbpx)
        print('seeds_ncc')
        print(ncc_error)
        print('pts: ' + str(qpt_ncc.shape[0]))
        print(ncc_fit_error)
        print('seeds_sbpx')
        print(sbpx_error)
        print('pts: ' + str(qpt_sbpx.shape[0]))
        print(sbpx_fit_error)
