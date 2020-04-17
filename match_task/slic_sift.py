import cv2 as cv
from match_task.common import *
import logging
import heapq
from multiprocessing import Pool, Manager, Lock


distance_thresh = 0.2
geometric_thresh = 12
ncc_templsize = 25

# expr_base = os.path.join(expr_base, 'rubust test')
map_path = os.path.join(data_dir, 'Image', 'Village0', 'map.jpg')
frame_dir = os.path.join(data_dir, 'Image', 'Village0', 'loc')
binary_dir = os.path.join(data_dir, 'Image', 'Village0', 'binary')
compactness = 16


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


def cv_match(queryIdx, trainIdx):
    match = cv.DMatch()
    match.distance = 0
    match.queryIdx = queryIdx
    match.trainIdx = trainIdx
    match.imgIdx = 0
    return match


def neighbors(point, x_limit, y_limit, width=3, height=3):
    assert width % 2 == 1 and height % 2 == 1
    neighbor_list = []
    x = int(point[0])
    y = int(point[1])
    for i in range(max(x - width // 2, x_limit[0]), min(x + width // 2 + 1, x_limit[1])):
        for j in range(max(y - height // 2, y_limit[0]), min(y + height // 2 + 1, y_limit[1])):
            if i != point[0] or j != point[1]:
                neighbor_list.append((i, j))
    return neighbor_list


def detect_compute_task(img, gray_img, x_scope, y_scope, content_mask, slic_mask,
                        corner_mask, out_pts_descs,calc_oriens,neighbor_refine):
    keypoints = []
    coord_set = {}
    corner_thred = corner_mask.max() * 0.01
    for y in range(y_scope[0], y_scope[1]):
        for x in range(x_scope[0], x_scope[1]):
            if content_mask[y][x] == 255 and (slic_mask[y][x] == 255 or corner_mask[y][x] > corner_thred):
                if calc_oriens:
                    coord = str(x) + ',' + str(y)
                    if coord not in coord_set.keys():
                        coord_set[coord] = ''
                        oriens = compute_orientation(gray_img, (x, y))
                        for orien in oriens:
                            keypoints.append(cv_point((x, y), orien))
                    for neighbor in neighbors((x, y), x_scope, y_scope, neighbor_refine,
                                              neighbor_refine):
                        coord = str(neighbor[0]) + ',' + str(neighbor[1])
                        if coord not in coord_set.keys():
                            coord_set[coord] = ''
                            oriens = compute_orientation(gray_img, neighbor)
                            for orien in oriens:
                                keypoints.append(cv_point(neighbor, orien))
                else:
                    coord = str(x) + ',' + str(y)
                    if coord not in coord_set.keys():
                        coord_set[coord] = ''
                        keypoints.append(cv_point((x, y)))
                    for neighbor in neighbors((x, y), x_scope, y_scope, neighbor_refine,
                                              neighbor_refine):
                        coord = str(neighbor[0]) + ',' + str(neighbor[1])
                        if coord not in coord_set.keys():
                            coord_set[coord] = ''
                            keypoints.append(cv_point(neighbor))
    detector = cv.xfeatures2d_SIFT.create()
    # points, descriptors = detector.compute(img, keypoints)
    points, descriptors = detector.compute(img, keypoints)
    out_pts_descs.append((points, descriptors))


def compute_task(img, detected_points, final_points, desc_arrays, lock):
    detector = cv.xfeatures2d_SIFT.create()
    # points, descriptors = detector.compute(img, keypoints)
    points, descriptors = detector.compute(img, detected_points)
    lock.acquire()
    final_points.append(points)
    desc_arrays.append(descriptors)
    lock.release()


def detect_compute(img, compactness=None, content_corners=None, draw=None, calc_oriens=False, neighbor_refine=1,
                   multi_p=4):
    # TODO use only one Pool
    if content_corners is None:
        content_corners = default_corners(img)
    if compactness is None:
        compactness = int((img.shape[0] * img.shape[1] / 1024) ** 0.5)
    lab_img = cvt2lab(img)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    slicer = cv.ximgproc.createSuperpixelSLIC(lab_img, region_size=compactness)
    slicer.iterate()
    slic_mask = slicer.getLabelContourMask()
    # corner detection
    corner_mask = cv.cornerHarris(gray_img, 2, 3, 0.04)
    content_mask = mask_of(img.shape[0], img.shape[1], content_corners)
    # keypoints = []
    # keypoints_dict = {}
    '''pool = Pool(multi_p)
    manager = Manager()
    pts_descs_tuples=manager.list([])
    x_part, y_part = img.shape[1] // multi_p + 1, img.shape[0] // multi_p + 1
    x_start, y_start = 0, 0
    for i in range(multi_p):
        x_end, y_end = min(img.shape[1], x_start + x_part), min(img.shape[0], y_start + y_part)
        pool.apply_async(detect_compute_task, args=(img, gray_img, (x_start, x_end), (y_start, y_end),
                                                    content_mask, slic_mask,corner_mask,pts_descs_tuples,
                                                    calc_oriens,neighbor_refine))
        x_start += x_part
        y_start += y_part
    pool.close()
    pool.join()
    points=[]
    desc_arrays=[]
    for item in pts_descs_tuples:
        points+=item[0]
        desc_arrays.append(item[1])
    descriptors=np.concatenate(desc_arrays,axis=0)'''
    keypoints = []
    coord_set = {}
    corner_thred = corner_mask.max() * 0.01
    corner_array = np.array(content_corners)
    x_max, x_min = corner_array[:, 0].max(), corner_array[:, 0].min()
    y_max, y_min = corner_array[:, 1].max(), corner_array[:, 1].min()
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if content_mask[y][x] == 255 and (slic_mask[y][x] == 255 or corner_mask[y][x] > corner_thred):
                if calc_oriens:
                    coord = str(x) + ',' + str(y)
                    if coord not in coord_set.keys():
                        coord_set[coord] = ''
                        oriens = compute_orientation(gray_img, (x, y))
                        for orien in oriens:
                            keypoints.append(cv_point((x, y), orien))
                    for neighbor in neighbors((x, y), (0, img.shape[1]), (0, img.shape[0]), neighbor_refine,
                                              neighbor_refine):
                        coord = str(neighbor[0]) + ',' + str(neighbor[1])
                        if coord not in coord_set.keys():
                            coord_set[coord] = ''
                            oriens = compute_orientation(gray_img, neighbor)
                            for orien in oriens:
                                keypoints.append(cv_point(neighbor, orien))
                else:
                    coord = str(x) + ',' + str(y)
                    if coord not in coord_set.keys():
                        coord_set[coord] = ''
                        keypoints.append(cv_point((x, y)))
                    for neighbor in neighbors((x, y), (0, img.shape[1]), (0, img.shape[0]), neighbor_refine,
                                              neighbor_refine):
                        coord = str(neighbor[0]) + ',' + str(neighbor[1])
                        if coord not in coord_set.keys():
                            coord_set[coord] = ''
                            keypoints.append(cv_point(neighbor))
    detector = cv.xfeatures2d_SIFT.create()
    # points, descriptors = detector.compute(img, keypoints)
    points, descriptors = detector.compute(img, keypoints)
    if draw is not None:
        copy = img.copy()
        corner_thred = corner_mask.max() * 0.01
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if content_mask[y][x] == 255:
                    if slic_mask[y][x] == 255:
                        if corner_mask[y][x] > corner_thred:
                            copy[y][x] = (255, 0, 0)
                        else:
                            copy[y][x] = (0, 0, 255)
                    elif corner_mask[y][x] > corner_thred:
                        copy[y][x] = (0, 255, 0)
        cv.imwrite(draw, copy)
    return points, descriptors


def feature_match(img1, img2, img1_features=None, img2_features=None, draw=None, match_result=None, sift_method=False):
    # max_matches=500):
    # neighbor_refine=1):
    # assert neighbor_refine > 0 and neighbor_refine % 2 == 1
    if img1_features is None:
        img1_point, img1_desc = detect_compute(img1)
    else:
        img1_point, img1_desc = img1_features[0], img1_features[1]
    if img2_features is None:
        img2_point, img2_desc = detect_compute(img2)
    else:
        img2_point, img2_desc = img2_features[0], img2_features[1]
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    if sift_method:
        ratio_thresh = 0.75
        raw_matches = matcher.knnMatch(img1_desc, img2_desc, 2)
        good_matches = []
        # match filtering
        corresponding1 = []
        corresponding2 = []
        for m, n in raw_matches:
            if m.distance < ratio_thresh * n.distance:
                # heapq.heappush(good_matches, (m.distance, m))
                good_matches.append(m)
                corresponding1.append(img1_point[m.queryIdx].pt)
                corresponding2.append(img2_point[m.trainIdx].pt)
        # use best 500-at-most matches
        ''' 
        if len(good_matches) <= max_matches:
            for match in good_matches:
                corresponding1.append(img1_point[match.queryIdx].pt)
                corresponding2.append(img2_point[match.trainIdx].pt)
        else:
            for i in range(max_matches):
                match = heapq.heappop(good_matches)
                corresponding1.append(img1_point[match.queryIdx].pt)
                corresponding2.append(img2_point[match.trainIdx].pt)
        '''
    else:
        raw_matches = matcher.knnMatch(img1_desc, img2_desc, 50)
        distance_valid_matches = raw_matches
        # distance threshoding
        # distance_valid_matches = []
        # for group in raw_matches:
        #     valid_group = []
        #     for match in group:
        #         if match.distance < distance_thresh:
        #             valid_group.append(match)
        #     if len(valid_group) > 0:
        #         distance_valid_matches.append(valid_group)
        # histogram voting
        histogram_voted_matches = []
        xtranslation_dict = {}
        ytranslation_dict = {}
        for group in distance_valid_matches:
            for match in group:
                point1 = img1_point[match.queryIdx]
                point2 = img2_point[match.trainIdx]
                xtranslation = point1.pt[0] - point2.pt[1]
                ytranslation = point1.pt[1] - point2.pt[1]
                xtranslation_dict.setdefault(xtranslation, 0)
                xtranslation_dict[xtranslation] += 1
                ytranslation_dict.setdefault(ytranslation, 0)
                ytranslation_dict[ytranslation] += 1
        voted_xtranslation = heapq.nlargest(1,
                                            [(translation, count) for translation, count in xtranslation_dict.items()],
                                            lambda item: item[1])
        voted_ytranslation = heapq.nlargest(1,
                                            [(translation, count) for translation, count in ytranslation_dict.items()],
                                            lambda item: item[1])
        for group in distance_valid_matches:
            valid_group = []
            for match in group:
                point1 = img1_point[match.queryIdx]
                point2 = img2_point[match.trainIdx]
                xtranslation = point1.pt[0] - point2.pt[1]
                ytranslation = point1.pt[1] - point2.pt[1]
                if abs(xtranslation - voted_xtranslation[0][0]) < geometric_thresh and abs(
                        ytranslation - voted_ytranslation[0][0]) < geometric_thresh:
                    valid_group.append(match)
            if len(valid_group) > 0:
                histogram_voted_matches.append(valid_group)
        # NCC match refinement
        corresponding1 = []
        corresponding2 = []
        # one_to_one_matches = []
        templ_expand = int((ncc_templsize - 1) / 2)
        search_expand = int((geometric_thresh - 1) / 2)
        for group in histogram_voted_matches:
            if len(group) == 1:
                # one_to_one_matches.append(group)
                corresponding1.append(img1_point[group[0].queryIdx].pt)
                corresponding2.append(img2_point[group[0].trainIdx].pt)
                continue
            query_point = np.int32(img1_point[group[0].queryIdx].pt)
            y_start = max(query_point[1] - templ_expand, 0)
            y_end = min(query_point[1] + templ_expand, img1.shape[0] - 1)
            x_start = max(query_point[0] - templ_expand, 0)
            x_end = min(query_point[0] + templ_expand, img1.shape[1] - 1)
            template = img1[y_start:y_end + 1, x_start:x_end + 1, :].copy()
            # 0 padding to template
            diff_ystart = query_point[1] - templ_expand
            diff_yend = query_point[1] + templ_expand - img1.shape[0] + 1
            diff_xstart = query_point[0] - templ_expand
            diff_xend = query_point[0] + templ_expand - img1.shape[1] + 1
            if diff_ystart < 0:
                template = np.concatenate([np.zeros((-diff_ystart, template.shape[1], 3)), template], axis=0)
            if diff_yend > 0:
                template = np.concatenate([template, np.zeros((diff_yend, template.shape[1], 3))], axis=0)
            if diff_xstart < 0:
                template = np.concatenate([np.zeros((template.shape[0], -diff_xstart, 3)), template], axis=1)
            if diff_xend > 0:
                template = np.concatenate([template, np.zeros((template.shape[0], diff_xend, 3))], axis=1)
            max_response = -1
            match_point = None
            for match in group:
                train_point = np.int32(img2_point[match.trainIdx].pt)
                search_window = (train_point[0], train_point[1], 2 * geometric_thresh + 1,
                                 2 * geometric_thresh + 1)  # center,width,height
                res_val, res_x, res_y = ncc_match(template, img2, search_window)
                if res_val > max_response:
                    match_point = (res_x, res_y)
            corresponding1.append((query_point[0], query_point[1]))
            corresponding2.append((match_point[0], query_point[1]))
    if match_result is not None:
        img1_points = []
        img2_points = []
        good_matches = []
        for i in range(len(corresponding1)):
            img1_points.append(cv_point(corresponding1[i]))
            img2_points.append(cv_point(corresponding2[i]))
            match = cv_match(i, i)
            good_matches.append(match)
        match_result.append((img1_points, img2_points, good_matches))
    if draw is not None:
        if match_result is None:
            match_result = []
            img1_points = []
            img2_points = []
            good_matches = []
            for i in range(len(corresponding1)):
                img1_points.append(cv_point(corresponding1[i]))
                img2_points.append(cv_point(corresponding2[i]))
                match = cv_match(i, i)
                good_matches.append(match)
            match_result.append((img1_points, img2_points, good_matches))
        else:
            img1_points, img2_points, matches = match_result[0]
            draw_match(img1, img1_points, img2, img2_points, matches, draw)
    return corresponding1, corresponding2


def ncc_match(template, img, search_window):
    # search_window is tuple (x,y,w,h)
    x, y, w, h = search_window
    w += template.shape[1] - 1
    h += template.shape[0] - 1
    h_expand = int((h - 1) / 2)
    w_expand = int((w - 1) / 2)
    y_start = int(max(y - h_expand, 0))
    y_end = int(min(y + h_expand, img.shape[0] - 1))
    x_start = int(max(x - w_expand, 0))
    x_end = int(min(x + w_expand, img.shape[1] - 1))
    search_img = img[y_start:y_end + 1, x_start:x_end + 1, :].copy()
    # 0 padding to template
    diff_ystart = y - h_expand
    diff_yend = y + h_expand - img.shape[0] + 1
    diff_xstart = x - w_expand
    diff_xend = x + w_expand - img.shape[1] + 1
    if diff_ystart < 0:
        search_img = np.concatenate([np.zeros((-diff_ystart, search_img.shape[1], 3)), search_img], axis=0)
    if diff_yend > 0:
        search_img = np.concatenate([search_img, np.zeros((diff_yend, search_img.shape[1], 3))], axis=0)
    if diff_xstart < 0:
        search_img = np.concatenate([np.zeros((search_img.shape[0], -diff_xstart, 3)), search_img], axis=1)
    if diff_xend > 0:
        search_img = np.concatenate([search_img, np.zeros((search_img.shape[0], diff_xend, 3))], axis=1)
    # template = cv.cvtColor(np.uint8(template), cv.COLOR_BGR2GRAY)
    # search_img = (cv.cvtColor(np.uint8(search_img), cv.COLOR_BGR2GRAY))
    response = cv.matchTemplate(np.uint8(search_img), np.uint8(template), method=cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(response)
    return max_val, max_loc[0] + x - w_expand, max_loc[1] + y - h_expand


def cvt2lab(img):
    img = cv.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    return img


def compute_orientation(gray_img, pt, sigma=1.5):
    # TODO consider acceleration
    # histogram voting
    neighbor_width = np.int32(np.ceil(sigma * 3))
    one_dim_kernel = cv.getGaussianKernel(2 * neighbor_width + 1, sigma)
    hist = np.zeros(36)
    kernel_left_top = (pt[0] - neighbor_width, pt[1] - neighbor_width)
    for dx in range(-neighbor_width, neighbor_width + 1):
        for dy in range(-neighbor_width, neighbor_width + 1):
            point_x, point_y = pt[0] + dx, pt[1] + dy
            if point_x < 0 or point_x > gray_img.shape[1] - 1:
                continue
            if point_y < 0 or point_y > gray_img.shape[0] - 1:
                continue
            mag, theta = gradient(gray_img, (point_x, point_y))
            bin = int(theta / 10)
            hist[bin] += mag * one_dim_kernel[point_x - kernel_left_top[0]][0] * one_dim_kernel[
                point_y - kernel_left_top[1]][0]
    # find dominat orientations
    orientations = []
    main_orientation = np.argmax(hist)
    max_voting = hist[main_orientation]
    orientations.append((main_orientation, max_voting))
    hist[main_orientation] = 0
    orien = np.argmax(hist)
    while hist[orien] > 0.8 * max_voting:
        orientations.append((orien, hist[orien]))
        hist[orien] = 0
    for item in orientations:
        hist[item[0]] = item[1]
    # fit accurate orientations
    accurate_oriens = []
    for item in orientations:
        centerval = item[0] * 10 + 5
        if item[0] == 35:
            rightval = 365
        else:
            rightval = centerval + 10
        if item[0] == 0:
            leftval = -5
        else:
            leftval = centerval - 10
        y = np.array([[centerval ** 2, centerval, 1],
                      [rightval ** 2, rightval, 1],
                      [leftval ** 2, leftval, 1]])
        x = np.array([hist[item[0]],
                      hist[(item[0] + 1) % 36],
                      hist[(item[0] - 1) % 36]])
        fit = np.linalg.lstsq(y, x, None)[0]
        if fit[0] == 0:
            fit[0] = 1e-6
        accurate_oriens.append(np.deg2rad(-x[1] / 2 * x[0]))
    return accurate_oriens


def gradient(gray_img, pt):
    x, y = pt
    dy = float(gray_img[min(gray_img.shape[0] - 1, y + 1), x]) - float(gray_img[max(0, y - 1), x])
    dx = float(gray_img[y, min(gray_img.shape[1] - 1, x + 1)]) - float(gray_img[y, max(0, x - 1)])
    magnitude = (dx ** 2 + dy ** 2) ** 0.5
    theta = np.arctan2(dy, dx) + np.pi
    deg = np.degrees(theta) % 360
    return magnitude, deg


def zero_orien_test():
    expr_dir = os.path.join(expr_base, 'zero_orien')
    reference = cv.imread(map_path)
    print('Map Load !')
    map_points, map_descs = map_features(reference)
    corners = pd.read_csv(os.path.join(frame_dir, 'corners.csv'))
    for frame_file in corners.index:
        corner = np.array([[corners.loc[frame_file, 'x1'], corners.loc[frame_file, 'y1'] + 5],
                           [corners.loc[frame_file, 'x2'] - 5, corners.loc[frame_file, 'y2']],
                           [corners.loc[frame_file, 'x0'] + 5, corners.loc[frame_file, 'y0']],
                           [corners.loc[frame_file, 'x3'], corners.loc[frame_file, 'y3'] - 5]])
        fileid = frame_file[:-4]
        frame = cv.imread(os.path.join(frame_dir, frame_file))
        match_result = []
        frame_points, frame_descs = detect_compute(frame, compactness,
                                                   draw=os.path.join(expr_dir, fileid + '_slic.png'),
                                                   content_corners=corner)
        print('Frame ' + fileid + ' features calculated !')
        img1_points, img2_points = feature_match(frame, reference, (frame_points, frame_descs), (map_points, map_descs),
                                                 os.path.join(expr_dir, fileid + '_slic_match.png'), match_result,
                                                 sift_method=True)
        print(fileid + ' match complete !')
        if len(img1_points) < 4 or len(img2_points) < 4:
            print(fileid + ' match failed !')
        else:
            img1_points = np.array(img1_points).reshape((-1, 1, 2))
            img2_points = np.array(img2_points).reshape((-1, 1, 2))
            retval, mask = homography(frame, reference, img1_points, img2_points,
                                      save_path=os.path.join(expr_dir, fileid + '_slic-homography.png'))
            if retval is None:
                print(fileid + ' slic sift match failed !')
            else:
                valid_matches = []
                for i in range(mask.shape[0]):
                    if mask[i][0] == 1:
                        valid_matches.append(match_result[0][2][i])
                print(fileid + ' valid matches: ' + str(len(valid_matches)))
                draw_match(frame, match_result[0][0], reference, match_result[0][1], valid_matches,
                           os.path.join(expr_dir, fileid + '_corrected_slic_match.png'))


def orien_test():
    expr_dir = os.path.join(expr_base, 'orien')
    reference = cv.imread(map_path)
    print('Map Load !')
    map_points, map_descs = map_features(reference, True)
    corners = pd.read_csv(os.path.join(frame_dir, 'corners.csv'))
    for frame_file in corners.index:
        corner = np.array([[corners.loc[frame_file, 'x1'], corners.loc[frame_file, 'y1'] + 5],
                           [corners.loc[frame_file, 'x2'] - 5, corners.loc[frame_file, 'y2']],
                           [corners.loc[frame_file, 'x0'] + 5, corners.loc[frame_file, 'y0']],
                           [corners.loc[frame_file, 'x3'], corners.loc[frame_file, 'y3'] - 5]])
        fileid = frame_file[:-4]
        frame = cv.imread(os.path.join(frame_dir, frame_file))
        match_result = []
        frame_points, frame_descs = detect_compute(frame, compactness,
                                                   draw=os.path.join(expr_dir, fileid + '_slic.png'),
                                                   content_corners=corner, calc_oriens=True)
        print('Frame ' + fileid + ' features calculated !')
        img1_points, img2_points = feature_match(frame, reference, (frame_points, frame_descs), (map_points, map_descs),
                                                 os.path.join(expr_dir, fileid + '_slic_match.png'), match_result,
                                                 sift_method=True)
        print(fileid + ' match complete !')
        if len(img1_points) < 4 or len(img2_points) < 4:
            print(fileid + ' match failed !')
        else:
            img1_points = np.array(img1_points).reshape((-1, 1, 2))
            img2_points = np.array(img2_points).reshape((-1, 1, 2))
            retval, mask = homography(frame, reference, img1_points, img2_points,
                                      save_path=os.path.join(expr_dir, fileid + '_slic-homography.png'))
            if retval is None:
                print(fileid + ' slic sift match failed !')
            else:
                valid_matches = []
                for i in range(mask.shape[0]):
                    if mask[i][0] == 1:
                        valid_matches.append(match_result[0][2][i])
                print(fileid + ' valid matches: ' + str(len(valid_matches)))
                draw_match(frame, match_result[0][0], reference, match_result[0][1], valid_matches,
                           os.path.join(expr_dir, fileid + '_corrected_slic_match.png'))


def demo_create():
    expr_dir = os.path.join(expr_base, 'demonstration')
    reference = cv.imread(map_path)
    print('Map Load !')
    map_points, map_descs = map_features(reference, True)
    corners = pd.read_csv(os.path.join(frame_dir, 'corners.csv'))
    for frame_file in corners.index:
        corner = np.array([[corners.loc[frame_file, 'x1'], corners.loc[frame_file, 'y1'] + 5],
                           [corners.loc[frame_file, 'x2'] - 5, corners.loc[frame_file, 'y2']],
                           [corners.loc[frame_file, 'x0'] + 5, corners.loc[frame_file, 'y0']],
                           [corners.loc[frame_file, 'x3'], corners.loc[frame_file, 'y3'] - 5]])
        fileid = frame_file[:-4]
        frame = cv.imread(os.path.join(frame_dir, frame_file))
        match_result = []
        frame_points, frame_descs = detect_compute(frame, compactness,
                                                   draw=os.path.join(expr_dir, fileid + '_slic.png'),
                                                   content_corners=corner, calc_oriens=True)
        print('Frame ' + fileid + ' features calculated !')
        img1_points, img2_points = feature_match(frame, reference, (frame_points, frame_descs), (map_points, map_descs),
                                                 os.path.join(expr_dir, fileid + '_slic_match.png'), match_result,
                                                 sift_method=True)
        print(fileid + ' match complete !')
        if len(img1_points) < 4 or len(img2_points) < 4:
            print(fileid + ' match failed !')
        else:
            img1_points = np.array(img1_points).reshape((-1, 1, 2))
            img2_points = np.array(img2_points).reshape((-1, 1, 2))
            retval, mask = homography(frame, reference, img1_points, img2_points,
                                      save_path=os.path.join(expr_dir, fileid + '_slic-homography.png'))
            if retval is None:
                print(fileid + ' slic sift match failed !')
            else:
                homo_corners = np.concatenate([corner, np.ones((4, 1))], axis=1)
                trans_corners = np.matmul(homo_corners, retval.T)
                trans_corners = np.int32(np.array(
                    [trans_corners[0][:-1] / trans_corners[0][-1], trans_corners[1][:-1] / trans_corners[1][-1],
                     trans_corners[3][:-1] / trans_corners[3][-1], trans_corners[2][:-1] / trans_corners[2][-1]]))
                poly_corners = trans_corners.reshape((-1, 1, 2))
                bgd = reference.copy()
                cv.polylines(bgd, poly_corners, isClosed=True, color=(0, 0, 255), thickness=18)
                # cv.line(bgd, tuple(trans_corners[0].tolist()), tuple(trans_corners[1].tolist()), color=(0, 0, 255),
                #         thickness=18)
                # cv.line(bgd, tuple(trans_corners[1].tolist()), tuple(trans_corners[2].tolist()), color=(0, 0, 255),
                #         thickness=18)
                # cv.line(bgd, tuple(trans_corners[2].tolist()), tuple(trans_corners[3].tolist()), color=(0, 0, 255),
                #         thickness=18)
                # cv.line(bgd, tuple(trans_corners[3].tolist()), tuple(trans_corners[0].tolist()), color=(0, 0, 255),
                #         thickness=18)
                cv.imwrite(os.path.join(expr_dir, frame_file), bgd)
                print(frame_file + ' complete !')


def scale_test(calc_orien=False, step=1):
    expr_dir = os.path.join(expr_base, 'scale')
    if calc_orien:
        expr_dir = os.path.join(expr_dir, 'calc_orien')
    else:
        expr_dir = os.path.join(expr_dir, 'zero_orien')
    logpath = os.path.join(expr_dir, 'match.log')
    logging.basicConfig(filename=logpath, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    reference = cv.imread(map_path)
    logger.info('Map Load !')
    map_points, map_descs = map_features(True)
    corners = pd.read_csv(os.path.join(frame_dir, 'corners.csv'))
    for i in range(0, len(corners.index), step):
        # for frame_file in corners.index:
        frame_file = corners.index[i]
        origin_corner = np.array([[corners.loc[frame_file, 'x1'], corners.loc[frame_file, 'y1'] + 5],
                                  [corners.loc[frame_file, 'x2'] - 5, corners.loc[frame_file, 'y2']],
                                  [corners.loc[frame_file, 'x0'] + 5, corners.loc[frame_file, 'y0']],
                                  [corners.loc[frame_file, 'x3'], corners.loc[frame_file, 'y3'] - 5]])
        fileid = frame_file[:-4]
        origin_frame = cv.imread(os.path.join(frame_dir, frame_file))
        for factor in range(5, 16):
            if factor == 10:
                continue
            factor /= 10.0
            frame, mat, corner = adaptive_scale(origin_frame, factor, content_corners=origin_corner)
            match_result = []
            frame_points, frame_descs = detect_compute(frame, compactness,
                                                       draw=os.path.join(expr_dir,
                                                                         fileid + '_factor' + str(
                                                                             factor) + '_slic.png'),
                                                       content_corners=corner, calc_oriens=calc_orien)
            logger.info('Frame ' + fileid + '_factor' + str(factor) + ' features calculated !')
            img1_points, img2_points = feature_match(frame, reference, (frame_points, frame_descs),
                                                     (map_points, map_descs),
                                                     os.path.join(expr_dir,
                                                                  fileid + '_factor' + str(factor) + '_slic_match.png'),
                                                     match_result, sift_method=True)
            logger.info(fileid + '_factor' + str(factor) + ' match complete !')
            if len(img1_points) < 4 or len(img2_points) < 4:
                logger.info(fileid + '_factor' + str(factor) + ' match failed !')
            else:
                img1_points = np.array(img1_points).reshape((-1, 1, 2))
                img2_points = np.array(img2_points).reshape((-1, 1, 2))
                retval, mask = homography(frame, reference, img1_points, img2_points,
                                          save_path=os.path.join(expr_dir,
                                                                 fileid + '_factor' + str(
                                                                     factor) + '_slic-homography.png'))
                if retval is None:
                    logger.info(fileid + '_factor' + str(factor) + ' slic sift match failed !')
                else:
                    valid_matches = []
                    for i in range(mask.shape[0]):
                        if mask[i][0] == 1:
                            valid_matches.append(match_result[0][2][i])
                    logger.info(fileid + '_factor' + str(factor) + ' valid matches: ' + str(len(valid_matches)))
                    draw_match(frame, match_result[0][0], reference, match_result[0][1], valid_matches,
                               os.path.join(expr_dir, fileid + '_factor' + str(factor) + '_corrected_slic_match.png'))


def rot_test(calc_orien=True, step=1):
    expr_dir = os.path.join(expr_base, 'rot')
    if calc_orien:
        expr_dir = os.path.join(expr_dir, 'calc_orien')
    else:
        expr_dir = os.path.join(expr_dir, 'zero_orien')
    logpath = os.path.join(expr_dir, 'match.log')
    logging.basicConfig(filename=logpath, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    reference = cv.imread(map_path)
    logger.info('Map Load !')
    map_points, map_descs = map_features(True)
    corners = pd.read_csv(os.path.join(frame_dir, 'corners.csv'))
    for i in range(0, len(corners.index), step):
        # for frame_file in corners.index:
        frame_file = corners.index[i]
        origin_corner = np.array([[corners.loc[frame_file, 'x1'], corners.loc[frame_file, 'y1'] + 5],
                                  [corners.loc[frame_file, 'x2'] - 5, corners.loc[frame_file, 'y2']],
                                  [corners.loc[frame_file, 'x0'] + 5, corners.loc[frame_file, 'y0']],
                                  [corners.loc[frame_file, 'x3'], corners.loc[frame_file, 'y3'] - 5]])
        fileid = frame_file[:-4]
        origin_frame = cv.imread(os.path.join(frame_dir, frame_file))
        for deg in range(-4, 5, 1):
            if deg == 0:
                continue
            deg *= 10
            frame, mat, corner = rotation_phi(origin_frame, deg, content_corners=origin_corner)
            match_result = []
            frame_points, frame_descs = detect_compute(frame, compactness,
                                                       draw=os.path.join(expr_dir,
                                                                         fileid + '_deg' + str(deg) + '_slic.png'),
                                                       content_corners=corner, calc_oriens=calc_orien)
            logger.info('Frame ' + fileid + '_deg' + str(deg) + ' features calculated !')
            img1_points, img2_points = feature_match(frame, reference, (frame_points, frame_descs),
                                                     (map_points, map_descs),
                                                     os.path.join(expr_dir,
                                                                  fileid + '_deg' + str(deg) + '_slic_match.png'),
                                                     match_result, sift_method=True)
            logger.info(fileid + '_deg' + str(deg) + ' match complete !')
            if len(img1_points) < 4 or len(img2_points) < 4:
                logger.info(fileid + '_deg' + str(deg) + ' match failed !')
            else:
                img1_points = np.array(img1_points).reshape((-1, 1, 2))
                img2_points = np.array(img2_points).reshape((-1, 1, 2))
                retval, mask = homography(frame, reference, img1_points, img2_points,
                                          save_path=os.path.join(expr_dir,
                                                                 fileid + '_deg' + str(deg) + '_slic-homography.png'))
                if retval is None:
                    logger.info(fileid + '_deg' + str(deg) + ' slic sift match failed !')
                else:
                    valid_matches = []
                    for i in range(mask.shape[0]):
                        if mask[i][0] == 1:
                            valid_matches.append(match_result[0][2][i])
                    logger.info(fileid + '_deg' + str(deg) + ' valid matches: ' + str(len(valid_matches)))
                    draw_match(frame, match_result[0][0], reference, match_result[0][1], valid_matches,
                               os.path.join(expr_dir, fileid + '_deg' + str(deg) + '_corrected_slic_match.png'))


def viewpoint_test():
    pass


def map_features(mapimg, calc_orien=False, with_corners=False, neighbors=1):
    if calc_orien:
        if with_corners:
            map_binary = os.path.join(binary_dir, 'map_features_calc_orien_n' + str(neighbors) + '_corners.pkl')
        else:
            map_binary = os.path.join(binary_dir, 'map_features_calc_orien_n' + str(neighbors) + '.pkl')
    else:
        if with_corners:
            map_binary = os.path.join(binary_dir, 'map_features_0orien_n' + str(neighbors) + '_corners.pkl')
        else:
            map_binary = os.path.join(binary_dir, 'map_features_0orien_n' + str(neighbors) + '.pkl')
    if not os.path.exists(map_binary):
        map_points, map_descs = detect_compute(mapimg, compactness, draw=os.path.join(expr_base, 'map_slic.png'),
                                               calc_oriens=calc_orien, neighbor_refine=neighbors)
        print('Map feartures calculated !')
        pt_vals = []
        for point in map_points:
            pt_vals.append(point_val(point))
        with open(map_binary, 'wb') as file:
            pickle.dump((pt_vals, map_descs), file)
        print('Map binary features saved !')
    else:
        with open(map_binary, 'rb') as file:
            feature = pickle.load(file)
        pt_vals, map_descs = feature
        map_points = []
        for val in pt_vals:
            map_points.append(cv_point(val[0], val[1]))
        print('Map binary features load !')
    return map_points, map_descs


def eval():
    import pandas as pd
    import logging

    base_dir = os.path.join(data_dir, 'Image', 'Village0')
    corrected_frame_dir = os.path.join(base_dir, 'loc')
    reference = cv.imread(map_path)
    print('Map Load !')
    map_points, map_descs = map_features(reference, True, True)
    corners = pd.read_csv(os.path.join(corrected_frame_dir, 'corners.csv'))
    correspondence = pd.read_csv(os.path.join(corrected_frame_dir, 'correspondence_points.csv'))
    expr_dir = os.path.join(expr_base, 'anno_frame_match')
    logging.basicConfig(filename=os.path.join(expr_dir, 'eval_log'), level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logging.basicConfig()
    frames = correspondence.index
    for frame_file in frames:
        logger.info(frame_file)
        fileid = frame_file[:-4]
        frame = cv.imread(os.path.join(corrected_frame_dir, frame_file))
        corner = np.array([[corners.loc[frame_file, 'x1'], corners.loc[frame_file, 'y1'] + 5],
                           [corners.loc[frame_file, 'x2'] - 5, corners.loc[frame_file, 'y2']],
                           [corners.loc[frame_file, 'x0'] + 5, corners.loc[frame_file, 'y0']],
                           [corners.loc[frame_file, 'x3'], corners.loc[frame_file, 'y3'] - 5]])
        test_list = []
        for i in range(1, 6):
            test_list.append([correspondence.loc[frame_file, 'x' + str(i) + '_1'],
                              correspondence.loc[frame_file, 'y' + str(i) + '_1']])
        target_list = []
        for i in range(1, 6):
            target_list.append([correspondence.loc[frame_file, 'x' + str(i) + '_2'],
                                correspondence.loc[frame_file, 'y' + str(i) + '_2']])
        match_result = []
        frame_points, frame_descs = detect_compute(frame, compactness,
                                                   draw=os.path.join(expr_dir, fileid + '_slic.png'),
                                                   content_corners=corner, calc_oriens=True, neighbor_refine=5)
        logger.info('Frame ' + fileid + ' features calculated !')
        img1_points, img2_points = feature_match(frame, reference, (frame_points, frame_descs), (map_points, map_descs),
                                                 os.path.join(expr_dir, fileid + '_slic_match.png'), match_result,
                                                 sift_method=True)
        logger.info(fileid + ' match complete !')
        if len(img1_points) < 4 or len(img2_points) < 4:
            logger.info(fileid + ' match failed !')
        else:
            img1_points = np.array(img1_points).reshape((-1, 1, 2))
            img2_points = np.array(img2_points).reshape((-1, 1, 2))
            retval, mask = homography(frame, reference, img1_points, img2_points, src_corners=corner,
                                      save_path=os.path.join(expr_dir, fileid + '_slic-homography.png'))
            logger.info('Homography: ')
            logger.info(str(retval))
            if retval is None:
                logger.info(fileid + ' slic sift match failed !')
            else:
                img1_points, img2_points, matches = match_result[0]
                match_test_list = []
                match_target_list = []
                valid_matches = []
                for i in range(mask.shape[0]):
                    if mask[i][0] == 1:
                        valid_matches.append(matches[i])
                        match_test_list.append(img1_points[matches[i].queryIdx].pt)
                        match_target_list.append(img2_points[matches[i].trainIdx].pt)
                logger.info(fileid + ' valid matches: ' + str(len(valid_matches)))
                draw_match(frame, match_result[0][0], reference, match_result[0][1], valid_matches,
                           os.path.join(expr_dir, fileid + '_corrected_slic_match.png'))

                def log_error(testlist, targetlist, homo_mat, log):
                    test_points_array = np.array(testlist)
                    log.info('Test points: ')
                    log.info(str(test_points_array))
                    homog_tests = np.concatenate([test_points_array, np.ones((len(testlist), 1))], axis=1)
                    homog_estimates = np.matmul(homog_tests, homo_mat.T)
                    homog_estimates /= homog_estimates[:, 2:]
                    target_array = np.array(targetlist)
                    estimate_array = homog_estimates[:, :-1]
                    log.info('Target points: ')
                    log.info(str(target_array))
                    log.info('Estimated points: ')
                    log.info(str(estimate_array))
                    square_error = (target_array - estimate_array) ** 2
                    point_wise_error = (np.sum(square_error, axis=-1)) ** 0.5
                    log.info('Point-wise error: ')
                    log.info(str(point_wise_error))
                    average_error = np.mean(point_wise_error, axis=-1)
                    log.info('Average error: ')
                    log.info(str(average_error))

                log_error(test_list, target_list, retval, logger)
                log_error(match_test_list, match_target_list, retval, logger)


if __name__ == '__main__':
    import pickle
    import pandas as pd
    from multiprocessing import Pool

    eval()
    '''demo_create()
    proc_pool = Pool(4)'''
    # orien_test()
    # zero_orien_test()

    # rot_test(calc_orien=True, step=3)
    '''
    proc_pool.apply_async(rot_test, args=(True, 3))
    print('calc_orien rot test submitted !')

    proc_pool.apply_async(scale_test, args=(True, 3))
    print('calc_orien scale test submitted !')

    proc_pool.apply_async(rot_test, args=(False, 3))
    print('zero_orien rot test submitted !')

    proc_pool.apply_async(scale_test, args=(False, 3))
    print('zero_orien scale test submitted !')

    proc_pool.close()
    proc_pool.join()
    print('Match complete !')'''
