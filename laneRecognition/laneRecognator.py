import numpy as np
import cv2

from birdsEye import BirdsEye
from laneFilter import LaneFilter
from curves import Curves
from util.util import roi


source_points = [(650, 450), (190, 720), (1250, 720), (870, 450)]
destination_points = [(330, 0), (320, 720), (960, 720), (960, 0)]

p = {'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
     'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}


class LaneRecognator:

    def __init__(self):
        self.birdsEye = BirdsEye()
        self.laneFilter = LaneFilter(p)
        self.curves = Curves(number_of_windows=9, margin=100, minimum_pixels=50,
                                ym_per_pix=30/720, xm_per_pix=3.7/700)

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