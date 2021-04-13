import threading

import cv2
import numpy as np

from events.dispatchers.refreshUiEventDispatcher import RefreshUiEventDispatcher
from util.util import get_random_RGB_colors, read_classes
from carDetection.yolo import Yolo
from laneRecognition.laneRecognator import LaneRecognator


class Detector:

    def draw_boxes(self, idxs, boxes, classIds, classes, confidences, frame):
        if len(idxs) == 0:
            return

        colors = get_random_RGB_colors()
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (width, height) = (boxes[i][2], boxes[i][3])

            color = [int(value) for value in colors[classIds[i]]]
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            text = '{}: {:.4f}'.format(classes[classIds[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def initialize_video_writer(self, videoStream, videoWidth, videoHeight, outputPath):
        sourceFPS = videoStream.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        return cv2.VideoWriter(outputPath, fourcc, sourceFPS,
                               (videoWidth, videoHeight), True)

    def detect(self, inputPath, outputPath,
               configPath='../config/yolov4-tiny.cfg',
               weightsPath='../config/yolov4-tiny.weights',
               classesPath='../config/coco-classes.txt',
               gpu=False):

        videoStream = cv2.VideoCapture(inputPath)

        if not videoStream.isOpened():
            print('Camera or video is not opened!')
            return

        videoWidth = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoHeight = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoWriter = self.initialize_video_writer(videoStream, videoWidth,
                                              videoHeight, outputPath)

        classes = read_classes(classesPath)
        yolo = Yolo(configPath, weightsPath, videoHeight, videoWidth, gpu)

        lane_recognator = LaneRecognator()
        refresh_ui_dispatcher = RefreshUiEventDispatcher()

        framesCount = 0

        while True:
            framesCount += 1
            print('FRAME ' + str(framesCount))

            (grabbed, frame) = videoStream.read()
            if not grabbed or getattr(threading.currentThread(), 'stop_process', False):
                break

            boxes, confidences, classIds = yolo.detect(frame)
            idxs = yolo.getNMSBoxes(boxes, confidences)

            frameForLane = np.copy(frame)
            lane_recognized_image, result = lane_recognator.pipeline(frameForLane)

            self.draw_boxes(idxs, boxes, classIds, classes, confidences, lane_recognized_image)

            refresh_ui_dispatcher.dispatch(result)

            # cv2.imshow('Frame', laneRecognizedImage)
            # videoWriter.write(laneRecognizedImage)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        print('Finished')

        videoStream.release()
        videoWriter.release()
        cv2.destroyAllWindows()
