from common import *
import siftmatch
import asiftmatch
import cv2 as cv

if __name__ == '__main__':
    map_path = os.path.join(data_dir, 'Image', 'Village0', 'original_map', 'map_village.jpg')
    frame_dir = os.path.join(data_dir, 'Image', 'Village0', 'original frames')
    # frame_dir = os.path.join(data_dir, 'Image')
    frame = cv.imread(os.path.join(frame_dir, 'loc3.JPG'))
    reference = cv.imread(map_path)
    '''corners = np.array([[0, 458], [613, 0], [
        612, 1278], [1226, 822]])'''
    '''augmented_frame, corners = data_augment(frame, expr_base, scale_factor=0.5)
    #match with augment
    # sift
    match_result = []
    img1_points, img2_points = siftmatch.feature_match(augmented_frame, reference, corners1=corners,
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
    # asift
    match_result = []
    img1_points, img2_points = asiftmatch.two_resolution_match(augmented_frame, reference, corners1=corners,
                                                               draw=os.path.join(expr_base, 'asift_match.png'),
                                                               match_result=match_result)
    retval, mask = homography(augmented_frame, reference, img1_points, img2_points, src_corners=corners,
                              save_path=os.path.join(expr_base, 'asift-homography.png'))
    if retval is None:
        print('augment asift match failed !')
    else:
        valid_matches = []
        for i in range(mask.shape[0]):
            if mask[i][0] == 1:
                valid_matches.append(match_result[0][2][i])
        draw_match(augmented_frame, match_result[0][0], reference, match_result[0][1], valid_matches,
                   os.path.join(expr_base, 'corrected_asift_match.png'))'''
    '''match with frame'''
    # sift
    '''match_result = []
    img1_points, img2_points = siftmatch.feature_match(frame, reference,
                                                       draw=os.path.join(
                                                           expr_base, 'sift_match', 'sift-match_origin.png'),
                                                       match_result=match_result)  # , corners1=corners)
    retval, mask = homography(frame, reference, img1_points, img2_points,
                              save_path=os.path.join(expr_base, 'sift_match', 'sift-homography_origin.png'),
                              ransac_iter=4096, ransac_thrd=1)
    if retval is None:
        print('origin sift match failed !')
    else:
        valid_matches = []
    for i in range(mask.shape[0]):
        if mask[i][0] == 1:
            valid_matches.append(match_result[0][2][i])
    draw_match(frame, match_result[0][0], reference, match_result[0][1], valid_matches,
               os.path.join(expr_base, 'sift_match', 'corrected_sift_match_origin.png'))'''
    # asift
    match_result = []
    img1_points, img2_points = asiftmatch.two_resolution_match(frame, reference,
                                                               draw=os.path.join(expr_base, 'asift_match',
                                                                                 'asift_match_origin.png'),
                                                               match_result=match_result)  # , corners1=corners)
    retval, mask = homography(frame, reference, img1_points, img2_points,
                              save_path=os.path.join(expr_base, 'asift_match', 'asift-homography_origin.png'),
                              ransac_iter=4096, ransac_thrd=1)
    if retval is None:
        print('origin asift match failed !')
    else:
        valid_matches = []
    for i in range(mask.shape[0]):
        if mask[i][0] == 1:
            valid_matches.append(match_result[0][2][i])
    draw_match(frame, match_result[0][0], reference, match_result[0][1], valid_matches,
               os.path.join(expr_base, 'asift_match','corrected_asift_match_origin.png'))
