import cv2


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

        cv2.imshow('Frame', frame)
        videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Finished')

    videoStream.release()
    videoWriter.release()
    cv2.destroyAllWindows()