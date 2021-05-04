import cv2
import numpy as np
from imageio import imread


class Chessboard:

    def __init__(self, path, x_number=9, y_number=6):
        self.path = path
        self.nx, self.ny = x_number, y_number
        self.gray_image = None
        self.dimensions = None
        self.obj_points, self.corners = None, None
        self.has_corners = False

    def initialize(self):
        image = imread(self.path)
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        rows, cols, _ = self.gray_image.shape
        self.dimensions = (rows, cols)

        self.obj_points = self.get_obj_points()
        self.has_corners, self.corners = self.find_chessboard_corners()

    def find_chessboard_corners(self):
        return cv2.findChessboardCorners(self.gray_image, (self.nx, self.ny), None)

    def get_obj_points(self):
        # (0, 0 ,0), (0, 1, 0)... (8, 5, 0)
        number_of_points = self.nx * self.ny
        points = np.zeros((number_of_points, 3), np.float32)
        points[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        return points