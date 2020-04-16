from cv2 import cv2 as cv
import numpy as np
import os
cwd = os.getcwd()
proj_dir=os.path.dirname(cwd)
common_dir = os.path.dirname(proj_dir)
data_dir = os.path.join(common_dir,'Datasets','UAVgeolocalization')
expr_base = os.path.join(proj_dir,'expriments','match_task')


def default_corners(img):
    return np.array([[0, 0], [img.shape[1], 0], [
        0, img.shape[0]], [img.shape[1], img.shape[0]]])


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


def draw_match(img1, img1_points, img2, img2_points, matches, path):
    result = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, img1_points, img2, img2_points, matches, result,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(path, result)


def homography(src_img, dst_img, src_points, dst_points, save_path, src_corners=None):
    if src_corners is None:
        src_corners = default_corners(src_img)
    retval, mask = cv.findHomography(src_points, dst_points, method=cv.RANSAC)
    if retval is None:
        return retval, mask
    # print(retval)
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
if __name__ == '__main__':
    origin_frame_dir = os.path.join(data_dir, 'Image', 'Village0', 'original frames')
    out_dir = os.path.join(data_dir, 'Image', 'Village0', 'anno_corrected_loc')
    files = os.listdir(origin_frame_dir)
    anno_corners = []
    for filename in files:
        img = cv.imread(os.path.join(origin_frame_dir, filename))
        scaled, scale_mat, scale_corners = adaptive_scale(img, 0.15)
        roted, rot_mat, rot_corners = rotation_phi(scaled, -132, scale_corners)
        cv.imwrite(os.path.join(out_dir, filename), roted)
        anno_corners.append(rot_corners)
    print(anno_corners)
    '''img = patch2background()
    cv.imwrite('./patch.jpg', img)'''
