from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox


class UiUtil:

    @staticmethod
    def show_message(message, title, priority):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setIcon(priority)
        msg.setText(title)
        msg.setInformativeText(message)
        msg.exec()

    @staticmethod
    def generate_output_video_path(directory, video_path):
        filename_with_extension = video_path.split('/')[-1]
        splitted_filename = filename_with_extension.split('.')
        return directory + '/' + splitted_filename[0] + '_output.avi' #+ splitted_filename[1]

    @staticmethod
    def arrimage2QPixmap(array):
        height, width, channel = array.shape
        bytesPerLine = 3 * width
        qImg = QImage(array.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return QPixmap(qImg)