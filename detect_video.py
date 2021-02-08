import cv2
import numpy as np
from util.util import getRandomRGDColors


def drawBoxes(idxs, boxes, classIds, confidences, frame):
    if len(idxs) == 0:
        return

    colors = getRandomRGDColors()
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (width, height) = (boxes[i][2], boxes[i][3])

        color = [int(value) for value in colors[classIds[i]]]
        cv2.rectangle(frame, (x, y), (x + width,y + height), color, 2)
        text = '{}: {:.4f}'.format('car', confidences[i])
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def initializeVideoWriter(videoStream, videoWidth, videoHeight, outputPath):
    sourceFPS = videoStream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    return cv2.VideoWriter(outputPath, fourcc, sourceFPS,
                           (videoWidth, videoHeight), True)


def initializeNet(configPath, weightsPath, useGPU):
    print('Loading YOLO')

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if useGPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, ln


def processLayerOutputs(layerOutputs, videoWidth, videoHeight, minConfidence=0.3):
    boxes, confidences, classIds = [], [], []

    for layerOutput in layerOutputs:
        for i, detection in enumerate(layerOutput):
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > minConfidence:
                box = detection[0:4] * np.array([videoWidth, videoHeight, videoWidth, videoHeight])
                (centerX, centerY, width, height) = box.astype('int')

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classID)

    return boxes, confidences, classIds


def detect(inputPath, outputPath, configPath, weightsPath, gpu=False):

    videoStream = cv2.VideoCapture(inputPath)

    if not videoStream.isOpened():
        print('Camera or video is not opened!')
        return

    videoWidth = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWriter = initializeVideoWriter(videoStream, videoWidth,
                                        videoHeight, outputPath)

    net, ln = initializeNet(configPath, weightsPath, gpu)

    framesCount = 0

    while True:
        framesCount += 1
        print('FRAME ' + str(framesCount))

        (grabbed, frame) = videoStream.read()
        if not grabbed:
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                    swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes, confidences, classIds = processLayerOutputs(layerOutputs, videoWidth, videoHeight)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        drawBoxes(idxs, boxes, classIds, confidences, frame)

        cv2.imshow('Frame', frame)
        videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Finished')

    videoStream.release()
    videoWriter.release()
    cv2.destroyAllWindows()