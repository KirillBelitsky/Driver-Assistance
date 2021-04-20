import threading
import time

import cv2
import numpy as np

from events.dispatchers.refreshUiEventDispatcher import RefreshUiEventDispatcher
from services.videoHelper import VideoHelper
from util.util import Util
from carDetection.yolo import Yolo
from laneRecognition.laneRecognator import LaneRecognator


class Detector:

    def __init__(self):
        self.yolo = None
        self.video_helper = None
        self.lane_recognator = LaneRecognator()
        self.refresh_ui_dispatcher = RefreshUiEventDispatcher()

    def draw_boxes(self, idxs, boxes, classIds, classes, confidences, frame):
        if len(idxs) == 0:
            return

        colors = Util.get_random_RGB_colors()
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (width, height) = (boxes[i][2], boxes[i][3])

            color = [int(value) for value in colors[classIds[i]]]
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            text = '{}: {:.4f}'.format(classes[classIds[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def calculate_fps(self, prev_time):
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        return fps, cur_time

    def detect(self, inputPath, outputPath,
               configPath='../config/yolov4-tiny.cfg',
               weightsPath='../config/yolov4-tiny.weights',
               classesPath='../config/coco-classes.txt',
               gpu=False):

        self.video_helper = VideoHelper(inputPath, outputPath)

        if not self.video_helper.is_video_stream_opened():
            print('Camera or video is not opened!')
            return

        classes = Util.read_classes(classesPath)

        video_size = self.video_helper.get_video_size()
        self.yolo = Yolo(configPath, weightsPath, video_size[1], video_size[0], gpu)

        frames_count = 0
        prev_frame_time = 0

        while True:

            (grabbed, frame) = self.video_helper.read_frame()
            if not grabbed or getattr(threading.currentThread(), 'stop_process', False):
                break

            boxes, confidences, classIds = self.yolo.detect(frame)
            idxs = self.yolo.get_NMS_boxes(boxes, confidences)

            frameForLane = np.copy(frame)
            lane_recognized_image, result = self.lane_recognator.debug_pipeline(frameForLane)

            self.draw_boxes(idxs, boxes, classIds, classes, confidences, lane_recognized_image)

            fps, prev_time = self.calculate_fps(prev_frame_time)
            result.update({'fps': fps})
            prev_frame_time = prev_time
            frames_count += 1
            print('FRAME: %s, FPS: %s' % (str(frames_count), str(np.round(fps, 2))))

            self.refresh_ui_dispatcher.dispatch(result)

            # cv2.imshow('Frame', lane_recognized_image)
            self.video_helper.write_frame(lane_recognized_image)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        print('Finished')

        self.video_helper.destroy_streams()
        cv2.destroyAllWindows()
