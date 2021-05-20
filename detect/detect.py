import argparse
from detector import Detector

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='path to input video')
parser.add_argument('--output', required=True, help='path to output video')
parser.add_argument('--config', required=False, help='path to yolo config files')
parser.add_argument('--classes', required=False, help='path to classes')
parser.add_argument('--weights', required=False, help='path to weights')
parser.add_argument('--gpu', required=False, help='GPU Cuda must be used (0 - no, 1 - yes')

args = vars(parser.parse_args())

input_path = args['input']
output_path = args['output']
weights_path = args['weights'] if args['weights'] is not None else 'config/yolov4-tiny.weights'
config_path = args['config'] if args['config'] is not None else 'config/yolov4-tiny.cfg'
classes_path = args['classes'] if args['classes'] is not None else 'config/coco-classes.txt'
gpu = True if args['gpu'] is not None and args['gpu'] == '1' else False

detector = Detector()
detector.detect(input_path, output_path, config_path, weights_path, classes_path, gpu)


