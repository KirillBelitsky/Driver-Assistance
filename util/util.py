import numpy as np
import cv2
import pickle


class Util:

    @staticmethod
    def RGB2HSL(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    @staticmethod
    def get_l_s_channels_from_HSL(image):
        return image[:, :, 1], image[:, :, 2]

    @staticmethod
    def get_random_RGB_colors(colorsAmount=100):
        np.random.seed(42)
        return np.random.randint(0, 255, size=(colorsAmount, 3), dtype="uint8")

    @staticmethod
    def read_classes(inputPath):
        with open(inputPath) as file:
            classNames = file.readlines()
        return [c.strip() for c in classNames]

    @staticmethod
    def get_calibration_properties(inputPath):
        calibration_data = pickle.load(open(inputPath, "rb"))
        matrix = calibration_data['camera_matrix']
        distortion_coef = calibration_data['distortion_coefficient']
        return matrix, distortion_coef

    @staticmethod
    def scale_abs(x, m=255):
        x = np.absolute(x)
        x = np.uint8(m * x / np.max(x))
        return x

    @staticmethod
    def roi(gray, mn=125, mx=1200):
        m = np.copy(gray) + 1
        m[:, :mn] = 0
        m[:, mx:] = 0
        return m
