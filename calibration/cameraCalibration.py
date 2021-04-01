import cv2
import pickle

from chessBoard import ChessBoard


# Initialize 20 chessboards
# note that at instatiation, it finds all chessboard corners and object points

chessboards = []

for n in range(20):
  this_path = '../images/calibrationImages/calibration' + str(n + 1) + '.jpg'
  chessboard = ChessBoard(i=n, path=this_path, nx=9, ny=6)
  chessboards.append(chessboard)


# We use these corners and object points (and image dimension)
# from all the chessboards to calculate the calibration parameters

points, corners, shape = [], [], chessboards[0].dimensions

for chessboard in chessboards:
  if chessboard.has_corners:
    points.append(chessboard.object_points)
    corners.append(chessboard.corners)

r, matrix, distortion_coef, rv, tv = cv2.calibrateCamera(points, corners, shape, None, None)


# Let's store these camera calibration parameters

calibration_data = {
    "camera_matrix": matrix,
    "distortion_coefficient": distortion_coef
}

pickle.dump(calibration_data, open("../config/calibration_data.p", "wb"))

