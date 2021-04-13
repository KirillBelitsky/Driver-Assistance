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