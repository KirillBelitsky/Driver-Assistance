import numpy as np
import cv2
from scipy.misc import imresize  # scipy 1.2.2

from laneRecognition.birdsEye import BirdsEye
from laneRecognition.laneFilter import LaneFilter
from laneRecognition.curves import Curves
from util.util import Util

#source_points = [(660, 450), (200, 720), (1250, 720), (870, 450)]
source_points = [(580, 460), (205, 720), (1110, 720), (703, 460)]
destination_points = [(320, 0), (320, 720), (960, 720), (960, 0)]

p = {'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
     'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}


class LaneRecognator:

    def __init__(self):
        matrix, distortion_coef = Util.get_calibration_properties('../config/calibration_data.p')

        self.birds_eye = BirdsEye(source_points, destination_points,
                                 matrix, distortion_coef)
        self.lane_filter = LaneFilter(p)
        self.curves = Curves(number_of_windows=9, margin=100, minimum_pixels=50,
                             ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)

    def pipeline(self, img):
        ground_img = self.birds_eye.undistort(img)
        binary = self.lane_filter.apply(ground_img)
        wb = np.logical_and(self.birds_eye.sky_view(binary), Util.roi(binary)).astype(np.uint8)
        result = self.curves.fit(wb)
        ground_img_with_projection = self.birds_eye.project(ground_img, binary,
                                                           result['pixel_left_best_fit_curve'],
                                                           result['pixel_right_best_fit_curve'])

        self.__add_addit_info(ground_img_with_projection, result, False)

        result.update({'result_image': ground_img_with_projection})

        return ground_img_with_projection, result

    def debug_pipeline(self, img):
        b_img, s_img, co_img, cu_img, pro_img, result = self.debug_pipeline_process(img)

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

        self.__add_addit_info(pro_img, result, True)

        result.update({'result_image': pro_img})

        return pro_img, result

    def debug_pipeline_process(self, img):
        ground_img = self.birds_eye.undistort(img)
        birdseye_img = self.birds_eye.sky_view(img)

        binary_img = self.lane_filter.apply(ground_img)
        sobel_img = self.birds_eye.sky_view(self.lane_filter.sobel_breakdown(ground_img))
        color_img = self.birds_eye.sky_view(self.lane_filter.color_breakdown(ground_img))

        wb = np.logical_and(self.birds_eye.sky_view(binary_img), Util.roi(binary_img)).astype(np.uint8)
        result = self.curves.fit(wb)

        left_curve = result['pixel_left_best_fit_curve']
        right_curve = result['pixel_right_best_fit_curve']

        curve_debug_img = result['image']

        projected_img = self.birds_eye.project(ground_img, binary_img, left_curve, right_curve)

        return birdseye_img, sobel_img, color_img, curve_debug_img, projected_img, result

    def __add_addit_info(self, img, result, is_debug):
        text_pos = "vehicle pos: " + result['vehicle_position_words']
        text_l = "left r: " + str(np.round(result['left_radius'], 2))
        text_r = " right r: " + str(np.round(result['right_radius'], 2))

        if not is_debug:
            cv2.putText(img, text_l, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, text_r, (400, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, text_pos, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(img, text_l, (20, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, text_r, (250, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, text_pos, (620, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)