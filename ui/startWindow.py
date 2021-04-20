import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog

from ui.detectVideoWindow import DetectVideoWindow
from ui.uiStartWindow import Ui_StartWindow
from util.uiutil import UiUtil


class StartWindow(QMainWindow):

    def __init__(self):
        super(StartWindow, self).__init__()
        self.ui = Ui_StartWindow()
        self.ui.setupUi(self)

        self.video_path = None
        self.output_video_path = None

        self.__init_button_listeners()

    def __init_button_listeners(self):
        self.ui.chooseVideoButton.clicked.connect(self.__on_click_chooseVideoButton)
        self.ui.chooseOutputPathButton.clicked.connect(self.__on_click_chooseOutputPathButton)
        self.ui.startButton.clicked.connect(self.__on_click_startButton)

    def __on_click_chooseVideoButton(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Choose video',
                                                  '/home/kirill/PythonProjects/Driver-Assistance/videos/',
                                                  'Videos (*.mp4 *.avi)')
        if filename:
            self.video_path = filename

    def __on_click_chooseOutputPathButton(self):
        if self.video_path is None:
            UiUtil.show_message('Firstly, choose video!', 'Error', QMessageBox.Critical)
            return

        directory = QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                     directory='/home/kirill/PythonProjects/Driver-Assistance/videos/output_videos/')
        if directory:
            self.output_video_path = UiUtil.generate_output_video_path(directory, self.video_path)

    def __on_click_startButton(self):
        validated, message = self.__validate()
        if not validated:
            UiUtil.show_message(message, 'Error', QMessageBox.Critical)
            return

        self.detectVideoWindow = DetectVideoWindow(self)
        self.detectVideoWindow.show()
        self.hide()
        self.detectVideoWindow.run_detection()

    def __validate(self):
        video_path_not_valid = self.video_path is None
        output_video_path_not_valid = self.output_video_path is None

        if video_path_not_valid and output_video_path_not_valid:
            return False, 'Please, choose required parameters!'
        elif video_path_not_valid \
                or output_video_path_not_valid:
            message = 'Please, choose video' if video_path_not_valid is None \
                else 'Please, choose output path'
            return False, message

        return True, None


if __name__ == '__main__':
    app = QApplication([])
    application = StartWindow()
    application.show()
    sys.exit(app.exec())
