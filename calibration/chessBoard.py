import cv2
import numpy as np
from imageio import imread


class ChessBoard:

    def __init__(self, i, path, nx=9, ny=6):
        self.path = path
        self.nx, self.ny = nx, ny

        temp_image = imread(self.path)
        temp_gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)

        rows, cols, channels = temp_image.shape
        self.dimensions = (rows, cols)

        self.has_corners, self.corners = cv2.findChessboardCorners(temp_gray, (self.nx, self.ny), None)
        self.object_points = self.get_object_points()

    def get_object_points(self):
        # (0, 0 ,0), (0, 1, 0)... (8, 5, 0)
        number_of_points = self.nx * self.ny
        points = np.zeros((number_of_points, 3), np.float32)
        points[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        return points