from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from events.dispatchers.eventDispatcher import EventDispatcher
from events.events.event import Event
from services.detectAdapter import DetectAdapter
from ui.uiDetectVideoWindow import Ui_DetectVideoWindow
from util.uiutil import UiUtil


class DetectVideoWindow(QMainWindow):

    def __init__(self, parentWindow):
        super(DetectVideoWindow, self).__init__()
        self.ui = Ui_DetectVideoWindow()
        self.ui.setupUi(self)

        self.detect_adapter = DetectAdapter()
        self.detect_adapter.set_callback(self.__refresh_data)

        self.event_dispatcher = EventDispatcher()

        self.event_dispatcher.register(Event.REFRESHUI, self.detect_adapter)
        self.event_dispatcher.register(Event.CLOSE, self.detect_adapter)

        self.parent = parentWindow
        self.__init_button_listeners()

    def __del__(self):
        self.event_dispatcher.unregister(Event.REFRESHUI, self.detect_adapter)
        self.event_dispatcher.unregister(Event.CLOSE, self.detect_adapter)

    def __init_button_listeners(self):
        self.ui.stopButton.clicked.connect(self.__on_click_stopButton)

    def __on_click_stopButton(self):
        UiUtil.show_message('Video will be written and window will be closed after closing notification!',
                            'Notification',
                            QMessageBox.Information)
        self.close()

    def run_detection(self):
        self.detect_adapter.detect(self.parent.video_path, self.parent.output_video_path)

    def __refresh_data(self, data):
        self.ui.image.setPixmap(data['pixmap'])
        self.ui.fpsValue.setText(data['fps'])
        self.ui.positionValue.setText(data['vehicle_pos'])
        self.ui.directionValue.setText(data['direction'])

    def closeEvent(self, event):
        self.event_dispatcher.dispatch(Event.CLOSE, None)
        self.__close_window(event)

    def __close_window(self, event):
        self.parent.show()
        event.accept()
        self.__del__()
