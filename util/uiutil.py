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