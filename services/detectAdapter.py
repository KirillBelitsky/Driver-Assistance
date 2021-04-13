from detector import Detector
import threading
import numpy as np

from events.listeners.closeEventListener import CloseEventListener
from events.listeners.refreshUiEventListener import RefreshUiEventListener
from util.uiutil import UiUtil

class DetectAdapter(CloseEventListener, RefreshUiEventListener):

    def __init__(self):
        self.detector = Detector()
        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def detect(self, video_path, output_video_path):
        self.x = threading.Thread(target=self.detector.detect,
                                  args=(video_path, output_video_path))
        self.x.start()

    def call(self, data):
        if data is None:
            self.x.stop_process = True
        else:
            if self.callback:
                self.callback(self.transform_data(data))

    def transform_data(self, data):
        image = data['result_image']
        pixmap = UiUtil.arrimage2QPixmap(image)

        result = {
            'pixmap': pixmap,
            'fps': str(np.round(data['fps'], 2)),
            'direction': data['direction'],
            'vehicle_pos': data['vehicle_position_words']
        }

        return result







