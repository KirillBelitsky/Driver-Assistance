import cv2
import numpy as np


class Yolo:

    def __init__(self, config_path, weights_path, video_height, video_width, use_GPU=True):
        self.config_path = config_path
        self.weights_path = weights_path
        self.use_GPU = use_GPU
        self.video_height = video_height
        self.video_width = video_width

        self.net = None
        self.ln = None

        self.initialize_net()

    def initialize_net(self):
        print('Loading YOLO')

        if self.weights_path.split('.')[-1] == 'weights':
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        else:
            self.net = cv2.dnn.readNetFromTensorflow('config/trained.pb')

        if self.use_GPU:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)

        return self.process_layer_outputs(layer_outputs)

    def process_layer_outputs(self, layer_outputs, min_confidence=0.3):
        boxes, confidences, classIds = [], [], []

        for layer_output in layer_outputs:
            for i, detection in enumerate(layer_output):
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > min_confidence:
                    box = detection[0:4] * np.array([self.video_width, self.video_height, self.video_width,  self.video_height])
                    (centerX, centerY, width, height) = box.astype('int')

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIds.append(classID)

        return boxes, confidences, classIds

    def get_NMS_boxes(self, boxes, confidences):
        return cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)