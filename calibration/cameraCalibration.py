import cv2
import pickle

from chessboard import Chessboard


# Initialize 20 chessboards
# note that at instatiation, it finds all chessboard corners and object points
class CameraCalibration:

    def __init__(self):
        self.chessboards = []

    def calibrate(self):
        for n in range(20):
            path = '../images/calibrationImages/calibration' + str(n + 1) + '.jpg'
            chessboard = Chessboard(path=path, x_number=9, y_number=6)
            self.chessboards.append(chessboard)

        points, corners, shape = self.get_data_needed_for_calibration()

        _, matrix, dist_coef, _, _ = cv2.calibrateCamera(points, corners, shape, None, None)

        data = {
            "camera_matrix": matrix,
            "distortion_coefficient": dist_coef
        }
        self.save_calibration_data(data)

    # We use these corners and object points (and image dimension)
    # from all the chessboards to calculate the calibration parameters
    def get_data_needed_for_calibration(self):
        points, corners, shape = [], [], self.chessboards[0].dimensions

        for chessboard in self.chessboards:
            if chessboard.has_corners:
                points.append(chessboard.object_points)
                corners.append(chessboard.corners)

        return points, corners, shape

    def save_calibration_data(self, calibration_data):
        pickle.dump(calibration_data, open("../config/calibration_data.p", "wb"))
