import cv2

def initializeVideoWriter(videoStream, videoWidth, videoHeight, outputPath):
    sourceFPS = videoStream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    return cv2.VideoWriter(outputPath, fourcc, sourceFPS,
                           (videoWidth, videoHeight), True)

def detect(inputPath, outputPath, configPath, weightsPath, gpu=False):

    print('Loading YOLO')

    videoStream = cv2.VideoCapture(inputPath)
    videoWidth = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWriter = initializeVideoWriter(videoStream, videoWidth,
                                        videoHeight, outputPath)

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