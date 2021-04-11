import numpy as np
import cv2
from scipy.misc import imresize  # scipy 1.2.2

from birdsEye import BirdsEye
from laneFilter import LaneFilter
from curves import Curves
from util.util import roi, getCalibrationProperties

source_points = [(650, 450), (190, 720), (1250, 720), (870, 450)]
destination_points = [(330, 0), (320, 720), (960, 720), (960, 0)]

p = {'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
     'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}


class LaneRecognator:

    def __init__(self):
        matrix, distortion_coef = getCalibrationProperties('../config/calibration_data.p')

        self.birdsEye = BirdsEye(source_points, destination_points,
                                 matrix, distortion_coef)
        self.laneFilter = LaneFilter(p)
        self.curves = Curves(number_of_windows=9, margin=100, minimum_pixels=50,
                             ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)

    def pipeline(self, img):
        ground_img = self.birdsEye.undistort(img)
        binary = self.laneFilter.apply(ground_img)
        wb = np.logical_and(self.birdsEye.sky_view(binary), roi(binary)).astype(np.uint8)
        result = self.curves.fit(wb)
        ground_img_with_projection = self.birdsEye.project(ground_img, binary,
                                                           result['pixel_left_best_fit_curve'],
                                                           result['pixel_right_best_fit_curve'])

        text_pos = "vehicle position: " + result['vehicle_position_words']
        text_l = "left radius: " + str(np.round(result['left_radius'], 2))
        text_r = " right radius: " + str(np.round(result['right_radius'], 2))

        cv2.putText(ground_img_with_projection, text_l, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(ground_img_with_projection, text_r, (400, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(ground_img_with_projection, text_pos, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        return ground_img_with_projection

    def debug_pipeline(self, img):
        b_img, s_img, co_img, cu_img, pro_img, lr, rr, pos = self.debug_pipeline_process(img)

        b_img = imresize(b_img, 0.25)
        s_img = imresize(s_img, 0.25)
        co_img = imresize(co_img, 0.25)
        cu_img = imresize(cu_img, 0.25)

        offset = [0, 320, 640, 960]
        width, height = 320, 180

        pro_img[:height, offset[0]: offset[0] + width] = b_img
        pro_img[:height, offset[1]: offset[1] + width] = co_img
        pro_img[:height, offset[2]: offset[2] + width] = s_img
        pro_img[:height, offset[3]: offset[3] + width] = cu_img

        text_pos = "vehicle pos: " + pos
        text_l = "left r: " + str(np.round(lr, 2))
        text_r = " right r: " + str(np.round(rr, 2))

        cv2.putText(pro_img, text_l, (20, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(pro_img, text_r, (250, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(pro_img, text_pos, (620, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        return pro_img

    def debug_pipeline_process(self, img):
        ground_img = self.birdsEye.undistort(img)
        birdseye_img = self.birdsEye.sky_view(img)

        binary_img = self.laneFilter.apply(ground_img)
        sobel_img = self.birdsEye.sky_view(self.laneFilter.sobel_breakdown(ground_img))
        color_img = self.birdsEye.sky_view(self.laneFilter.color_breakdown(ground_img))

        wb = np.logical_and(self.birdsEye.sky_view(binary_img), roi(binary_img)).astype(np.uint8)
        result = self.curves.fit(wb)

        left_curve = result['pixel_left_best_fit_curve']
        right_curve = result['pixel_right_best_fit_curve']

        left_radius = result['left_radius']
        right_radius = result['right_radius']
        pos = result['vehicle_position_words']
        curve_debug_img = result['image']

        projected_img = self.birdsEye.project(ground_img, binary_img, left_curve, right_curve)

        return birdseye_img, sobel_img, color_img, curve_debug_img, projected_img, left_radius, right_radius, pos