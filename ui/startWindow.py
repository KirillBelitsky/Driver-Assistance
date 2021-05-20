import sys
import mimetypes

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
        mimetypes.init()

    def __init_button_listeners(self):
        self.ui.chooseVideoButton.clicked.connect(self.__on_click_chooseVideoButton)
        self.ui.chooseOutputPathButton.clicked.connect(self.__on_click_chooseOutputPathButton)
        self.ui.startButton.clicked.connect(self.__on_click_startButton)

    def __on_click_chooseVideoButton(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Выберите видеофайл',
                                                  '/home/kirill/PythonProjects/Driver-Assistance/videos/')
        if filename:
            self.video_path = filename

    def __on_click_chooseOutputPathButton(self):
        if self.video_path is None:
            UiUtil.show_message('Сперва, выберите видеофайл!', 'Ошибка', QMessageBox.Critical)
            return

        directory = QFileDialog.getExistingDirectory(self, caption='Выберите директорию',
                                                     directory='/home/kirill/PythonProjects/Driver-Assistance/videos/output_videos/')
        if directory:
            self.output_video_path = UiUtil.generate_output_video_path(directory, self.video_path)

    def __on_click_startButton(self):
        validated, message = self.__validate()
        if not validated:
            UiUtil.show_message(message, 'Ошибка', QMessageBox.Critical)
            return

        self.detectVideoWindow = DetectVideoWindow(self)
        self.detectVideoWindow.show()
        self.hide()
        self.detectVideoWindow.run_detection()

    def __validate(self):
        video_path_not_valid = self.video_path is None
        output_video_path_not_valid = self.output_video_path is None

        if video_path_not_valid and output_video_path_not_valid:
            return False, 'Пожалуйста, выберите видеофайл и директорию!'
        elif video_path_not_valid \
                or output_video_path_not_valid:
            message = 'Пожалуйста, выберите видеофайл!' if video_path_not_valid is None \
                else 'Пожалуйста, выберите директорию!'
            return False, message

        mimestart = mimetypes.guess_type(self.video_path)[0]
        if mimestart is not None:
            mimestart = mimestart.split('/')[0]

            if mimestart != 'video':
                return False, 'Выбранный файл не является видеофайлом!'

        return True, None


if __name__ == '__main__':
    app = QApplication([])
    application = StartWindow()
    application.show()
    sys.exit(app.exec())
