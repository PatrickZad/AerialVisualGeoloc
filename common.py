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
    # homo_mat is h*3*3
    # test_pts and target_pts are n*k*2
    assert len(homo_mat.shape) == len(test_pts.shape) == len(target_pts.shape) == 3
    homog_tests = np.concatenate([test_pts, np.ones((test_pts.shape[0], test_pts.shape[1], 1))], axis=-1)
    homog_estimates = np.matmul(homog_tests, np.transpose(homo_mat, (0, 2, 1)))
    homog_estimates /= homog_estimates[..., -1:]
    estimate_array = homog_estimates[..., :-1]
    square_error = (target_pts - estimate_array) ** 2
    point_wise_error = (np.sum(square_error, axis=-1)) ** 0.5
    average_error = np.mean(point_wise_error, axis=-1)
    if log is not None:
        log.info('Test points: ')
        log.info(str(test_pts))
        log.info('Target points: ')
        log.info(str(target_pts))
        log.info('Estimated points: ')
        log.info(str(estimate_array))
        log.info('Point-wise error: ')
        log.info(str(point_wise_error))
        log.info('Average error: ')
        log.info(str(average_error))


if __name__ == '__main__':
    import pandas as pd
    import logging

    logging.basicConfig(filename=os.path.join(expr_base, 'eval', 'handpick_eval'), level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    '''center locs'''
    loc_data_dir = os.path.join(data_dir, 'Image', 'Village0', 'crop_loc', 'center_crops')
    loc_ids = list(range(0, 17, 2))
    # crop_subpx_0orien_withdrop
    retvals0 = np.array([
        [[-6.26324013e+00, -5.02172304e+00, 4.55360656e+03],
         [-2.01195326e+00, -1.08854429e+00, 1.36386045e+03],
         [-1.44977486e-03, -1.07000405e-03, 1.00000000e+00]],
        [[1.12362459e+00, - 1.97120551e+00, 4.10692227e+03],
         [1.48817242e-01, 2.64788642e-02, 1.56762950e+03],
         [7.69191332e-05, - 4.55472619e-04, 1.00000000e+00]],
        [[1.10871803e-01, 3.62364815e-01, 3.89749336e+03],
         [-3.55495795e-01, 1.02075383e+00, 1.90401268e+03],
         [-1.78562832e-04, 8.82267596e-05, 1.00000000e+00]],
        [[2.17575825e-01, - 9.00028319e-01, 3.59268736e+03],
         [-4.27675486e-01, 2.60014268e-01, 2.11563883e+03],
         [-1.53750426e-04, - 2.54386397e-04, 1.00000000e+00]],
        [[5.69951359e-01, - 6.98398429e-01, 3.29303454e+03],
         [-1.91646176e-01, 3.39167187e-01, 2.33881419e+03],
         [-7.39155719e-05, - 2.03175481e-04, 1.00000000e+00]],
        [[-3.16856181e-01, - 2.34686125e+00, 3.01760490e+03],
         [-8.58748448e-01, - 1.33439529e+00, 2.60683603e+03],
         [-3.09612011e-04, - 7.41912304e-04, 1.00000000e+00]],
        [[2.06384704e+00, - 1.47154930e+00, 2.69788757e+03],
         [1.35316383e+00, - 7.84333659e-01, 2.88954268e+03],
         [4.70499146e-04, - 5.29556560e-04, 1.00000000e+00]],
        [[4.63902647e-01, - 1.08299252e+00, 2.43625136e+03],
         [-2.97210965e-01, - 6.49732420e-01, 3.14980329e+03],
         [-9.83142290e-05, - 4.21104913e-04, 1.00000000e+00]],
        [[6.83897879e-01, - 1.20858853e+00, 2.20271294e+03],
         [-1.04756516e-02, - 1.00886701e+00, 3.46522738e+03],
         [-2.20378241e-05, - 4.95748888e-04, 1.00000000e+00]]
    ])
    # crop_subpx_0orien_nodrop
    retvals1 = np.array([
        [[-7.23835376e+00, 1.19727330e+00, 4.53926506e+03],
         [-2.32638099e+00, 9.35458606e-01, 1.34891634e+03],
         [-1.67607733e-03, 2.70778784e-04, 1.00000000e+00]],
        [[1.09024558e+00, - 4.91935467e-01, 4.09398632e+03],
         [9.72883059e-02, 7.07220511e-01, 1.55508642e+03],
         [5.24063161e-05, - 1.06274425e-04, 1.00000000e+00]],
        [[1.68211018e+00, - 2.09661962e+00, 3.90152755e+03],
         [3.99256420e-01, - 1.67566377e-01, 1.90463552e+03],
         [2.15783575e-04, - 5.18444446e-04, 1.00000000e+00]],
        [[-4.95988726e-01, - 2.52189035e+00, 3.61169295e+03],
         [-7.43996543e-01, - 8.59010384e-01, 2.12983489e+03],
         [-3.15225374e-04, - 6.90181909e-04, 1.00000000e+00]],
        [[8.26728062e-01, - 6.41500722e-01, 3.28944183e+03],
         [-2.69799872e-02, 4.10897310e-01, 2.33538264e+03],
         [-5.54290364e-06, - 1.86582997e-04, 1.00000000e+00]],
        [[-4.20631725e-01, - 2.97055782e+00, 3.02013855e+03],
         [-9.20292862e-01, - 1.86459494e+00, 2.60260924e+03],
         [-3.35463909e-04, - 9.44133959e-04, 1.00000000e+00]],
        [[-4.04062152e-01, - 2.23077065e+00, 2.71542009e+03],
         [-1.16036115e+00, - 1.80326389e+00, 2.92798330e+03],
         [-3.56404107e-04, - 8.01721182e-04, 1.00000000e+00]],
        [[7.05442365e-01, - 1.62562229e+00, 2.44286363e+03],
         [4.97628851e-02, - 1.37830263e+00, 3.15807355e+03],
         [1.51858532e-06, - 6.25866924e-04, 1.00000000e+00]],
        [[1.36308119e+00, 1.04056568e+00, 2.11843220e+03],
         [5.08940141e-01, 2.74105287e+00, 3.41839622e+03],
         [1.29634562e-04, 4.49927411e-04, 1.00000000e+00]]
    ])
    # crop_0orien_nmap_corner_unit
    retvals2 = np.array([
        [[-3.63778594e+00, - 1.67753707e+00, 4.52217746e+03],
         [-1.20718808e+00, 1.13526457e-01, 1.33364703e+03],
         [-9.18523504e-04, - 3.49508635e-04, 1.00000000e+00]],
        [[3.62554234e+00, 1.62074279e+01, 3.91276902e+03],
         [8.79179466e-01, 8.16139002e+00, 1.40158152e+03],
         [4.53659604e-04, 3.79741681e-03, 1.00000000e+00]],
        [[7.45780101e-01, - 1.63506864e+00, 3.90159959e+03],
         [-1.13929549e-02, 2.27892737e-03, 1.90646884e+03],
         [-9.46500226e-06, - 4.09419852e-04, 1.00000000e+00]],
        [[1.41681890e+00, - 1.48578575e+00, 3.59778106e+03],
         [3.03351611e-01, - 2.58300413e-02, 2.10338146e+03],
         [1.65015967e-04, - 4.03437852e-04, 1.00000000e+00]],
        [[3.43860870e-01, 1.58566622e-01, 3.28771191e+03],
         [-3.71866114e-01, 9.70529318e-01, 2.33699917e+03],
         [-1.41437870e-04, 3.59043920e-05, 1.00000000e+00]],
        [[-3.03836388e-01, - 1.73288532e+00, 3.01332411e+03],
         [-8.52095534e-01, - 7.57996314e-01, 2.59834751e+03],
         [-3.11073127e-04, - 5.45452579e-04, 1.00000000e+00]],
        [[-2.76892461e-01, 1.27323466e+00, 2.68968675e+03],
         [-1.29924258e+00, 2.39785562e+00, 2.85600726e+03],
         [-4.07627908e-04, 4.63734268e-04, 1.00000000e+00]],
        [[3.84134076e-01, - 1.06676626e+00, 2.44227856e+03],
         [-4.15734393e-01, - 5.76830859e-01, 3.14536035e+03],
         [-1.33342908e-04, - 4.04595185e-04, 1.00000000e+00]],
        [[3.58668337e-01, - 5.54727385e-01, 2.31496140e+03],
         [-5.81087727e-01, 2.31200920e-02, 3.30876125e+03],
         [-1.76365672e-04, - 2.26698384e-04, 1.00000000e+00]]
    ])
    # crop_0orien_nmap_corner
    retvals3 = np.array([
        [[-8.40145058e+00, - 3.68110217e+00, 4.54900623e+03],
         [-2.65534661e+00, - 6.93862040e-01, 1.36183611e+03],
         [-1.90983673e-03, - 7.90611257e-04, 1.00000000e+00]],
        [[1.20076737e+00, - 1.22883444e+00, 4.09962230e+03],
         [1.63761549e-01, 3.55965468e-01, 1.56289515e+03],
         [8.73869314e-05, - 2.82063214e-04, 1.00000000e+00]],
        [[-1.78039287e-01, - 7.98505577e-01, 3.90020163e+03],
         [-4.81405277e-01, 4.14493089e-01, 1.90752780e+03],
         [-2.38945995e-04, - 2.02983533e-04, 1.00000000e+00]],
        [[-7.16093962e-01, - 1.52286597e+00, 3.60637193e+03],
         [-9.13664412e-01, - 2.00778928e-01, 2.12370185e+03],
         [-3.83695193e-04, - 4.20916201e-04, 1.00000000e+00]],
        [[3.68737951e-01, - 3.95183205e-01, 3.29377445e+03],
         [-3.35969580e-01, 5.59153760e-01, 2.33957009e+03],
         [-1.33061764e-04, - 1.10670155e-04, 1.00000000e+00]],
        [[8.62997287e-01, - 7.70430015e-01, 2.97055397e+03],
         [3.24461114e-02, 9.24421979e-02, 2.57942611e+03],
         [1.41067640e-05, - 2.70593726e-04, 1.00000000e+00]],
        [[6.88638561e-02, - 9.22417776e-01, 2.68743032e+03],
         [-7.82727494e-01, - 3.09374081e-01, 2.90494227e+03],
         [-2.32511512e-04, - 3.51208690e-04, 1.00000000e+00]],
        [[1.64441165e-01, - 8.85974822e-01, 2.43388224e+03],
         [-7.04188425e-01, - 3.72871285e-01, 3.14465792e+03],
         [-2.17399303e-04, - 3.45276127e-04, 1.00000000e+00]],
        [[5.35150866e-02, - 1.47978008e+00, 2.23063732e+03],
         [-8.88751382e-01, - 1.52032319e+00, 3.51100338e+03],
         [-2.48147691e-04, - 6.01628045e-04, 1.00000000e+00]]
    ])
    retvals = {'crop_subpx_0orien_withdrop': retvals0, 'crop_subpx_0orien_nodrop': retvals1,
               'crop_0orien_nmap_corner_unit': retvals2, 'crop_0orien_nmap_corner': retvals3}
    for desc, retval_array in retvals.items():
        test_pts = []
        target_pts = []
        for idx in loc_ids:
            csv_file = os.path.join(loc_data_dir, 'center_loc' + str(idx) + '.csv')
            df = pd.read_csv(csv_file)
            test_array = np.array([
                center_of_corners(np.reshape(df['pt1', 'loc_x1':'loc_y4'], (4, 2))),
                center_of_corners(np.reshape(df['pt2', 'loc_x1':'loc_y4'], (4, 2))),
                center_of_corners(np.reshape(df['pt3', 'loc_x1':'loc_y4'], (4, 2))),
                center_of_corners(np.reshape(df['pt4', 'loc_x1':'loc_y4'], (4, 2)))
            ])
            test_pts.append(test_array)
            target_array = np.array([
                center_of_corners(np.reshape(df['pt1', 'map_x1':'map_y4'], (4, 2))),
                center_of_corners(np.reshape(df['pt2', 'mao_x1':'map_y4'], (4, 2))),
                center_of_corners(np.reshape(df['pt3', 'map_x1':'map_y4'], (4, 2))),
                center_of_corners(np.reshape(df['pt4', 'map_x1':'map_y4'], (4, 2))),
            ])
            target_pts.append(target_array)
        logger.info(desc)
        warp_error(test_pts, target_pts, retval_array, logger)
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
