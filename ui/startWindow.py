import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog

from ui.mainWindow import MainWindow
from ui.uiStartWindow import Ui_StartWindow


class StartWindow(QMainWindow):

    def __init__(self):
        super(StartWindow, self).__init__()
        self.ui = Ui_StartWindow()
        self.ui.setupUi(self)

        self.init_button_listeners()

    def init_button_listeners(self):
        self.ui.chooseVideoButton.clicked.connect(self.on_click_chooseVideoButton)
        self.ui.chooseOutputPathButton.clicked.connect(self.on_click_chooseOutputPathButton)
        self.ui.startButton.clicked.connect(self.on_click_startButton)

    def on_click_chooseVideoButton(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Choose video',
                                                  '/home/kirill/PythonProjects/Driver-Assistance/videos/',
                                                  'Videos (*.mp4 *.avi)')
        if filename:
            self.ui.videoPath = filename

    def on_click_chooseOutputPathButton(self):
        if self.ui.videoPath is None:
            self.show_error_message('Firstly, choose video!')
            return

        directory = QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                     directory='/home/kirill/PythonProjects/Driver-Assistance/videos/output_videos/')
        if directory:
            filename_with_extension = self.ui.videoPath.split('/')[-1]
            splitted_filename = filename_with_extension.split('.')
            self.ui.outputVideoPath = directory + '/' + splitted_filename[0] + '_output.' + splitted_filename[1]

    def on_click_startButton(self):
        validated, message = self.validate()
        if not validated:
            self.show_error_message(message)
            return

        self.mainWindow = MainWindow(self)
        self.mainWindow.show()
        self.hide()

    def validate(self):
        video_path_not_valid = self.ui.videoPath is None
        output_video_path_not_valid = self.ui.outputVideoPath is None

        if video_path_not_valid and output_video_path_not_valid:
            return False, 'Please, choose required parameters!'
        elif video_path_not_valid is None \
                or output_video_path_not_valid is None:
            message = 'Please, choose video' if video_path_not_valid is None \
                else 'Please, choose output path'
            return False, message

        return True, None

    def show_error_message(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setIcon(QMessageBox.Critical)
        msg.setText('Error')
        msg.setInformativeText(message)
        msg.exec()


if __name__ == '__main__':
    app = QApplication([])
    application = StartWindow()
    application.show()
    sys.exit(app.exec())
