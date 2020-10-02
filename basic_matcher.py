import cv2 as cv
import os
import common


class BasicMatcher:
    def __init__(self, dc, m, e):
        self.detect_compute = dc
        self.match = m
        self.estimate = e

    def match_estimate(self, ref, tar, match_vis=None, gt_match=None):
        ref_kps, ref_features = self.detect_compute(ref)  # iterable collection of (x,y)-coordinates
        tar_kps, tar_features = self.detect_compute(tar)
        correspondencs = self.match(ref_kps, ref_features, tar_kps, tar_kps)
        homog, filtered_corres = self.estimate(correspondencs)
        result = homog
        if match_vis is not None:
            common.match_vis(ref, tar, filtered_corres, match_vis)
        if gt_match is not None:
            error = common.estimate_error(homog, gt_match)
            result = [homog, error]
        return result


def sift_detect(img_arr=None):
    pass
