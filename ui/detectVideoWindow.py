from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from events.dispatchers.closeEventDispatcher import CloseEventDispatcher
from events.dispatchers.refreshUiEventDispatcher import RefreshUiEventDispatcher
from services.detectAdapter import DetectAdapter
from ui.uiDetectVideoWindow import Ui_DetectVideoWindow
from util.uiutil import UiUtil


class DetectVideoWindow(QMainWindow):

    def __init__(self, parentWindow):
        super(DetectVideoWindow, self).__init__()
        self.ui = Ui_DetectVideoWindow()
        self.ui.setupUi(self)

        self.detectAdapter = DetectAdapter()
        self.detectAdapter.set_callback(self.refresh_data)

        self.closeEventDispatcher = CloseEventDispatcher()
        self.refreshUiEventDispatcher = RefreshUiEventDispatcher()

        self.closeEventDispatcher.register(self.detectAdapter)
        self.refreshUiEventDispatcher.register(self.detectAdapter)

        self.parent = parentWindow
        self.init_button_listeners()

    def __del__(self):
        self.closeEventDispatcher.unregister(self.detectAdapter)
        self.refreshUiEventDispatcher.unregister(self.detectAdapter)

    def init_button_listeners(self):
        self.ui.stopButton.clicked.connect(self.on_click_stopButton)

    def on_click_stopButton(self):
        UiUtil.show_message('Video will be written and window will be closed after closing notification!',
                            'Notification',
                            QMessageBox.Information)
        self.close()

    def run_detection(self):
        self.detectAdapter.detect(self.parent.ui.videoPath, self.parent.ui.outputVideoPath)

    def refresh_data(self, data):
        self.ui.image.setPixmap(data['pixmap'])
        self.ui.fpsValue.setText(data['fps'])
        self.ui.positionValue.setText(data['vehicle_pos'])
        self.ui.directionValue.setText(data['direction'])

    def closeEvent(self, event):
        self.closeEventDispatcher.dispatch(None)
        self.close_window(event)

    def close_window(self, event):
        self.parent.show()
        event.accept()
        self.__del__()
