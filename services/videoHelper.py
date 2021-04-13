import cv2


class VideoHelper:

    def __init__(self, input_video_path, output_video_path):
        self.video_stream = None
        self.video_writer = None

        self.video_width = None
        self.video_height = None

        self.init_video_stream(input_video_path)
        self.initialize_video_writer(output_video_path)

    def init_video_stream(self, input_video_path):
        self.video_stream = cv2.VideoCapture(input_video_path)

        self.video_width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def initialize_video_writer(self, output_path):
        sourceFPS = self.video_stream.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        self.video_writer = cv2.VideoWriter(output_path, fourcc, sourceFPS,
                                            (self.video_width, self.video_height), True)

    def get_video_size(self):
        return (self.video_width, self.video_height)

    def is_video_stream_opened(self):
        return self.video_stream.isOpened()

    def read_frame(self):
        return self.video_stream.read()

    def write_frame(self, frame):
        self.video_writer.write(frame)

    def destroy_streams(self):
        self.video_stream.release()
        self.video_writer.release()
