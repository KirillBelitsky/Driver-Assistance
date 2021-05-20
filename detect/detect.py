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
classes_path = args['classes']
weights_path = args['weights']
config_path = args['config']
gpu = args['gpu']
gpu = True if gpu is not None and gpu == '1' else False

detector = Detector()
detector.detect(input_path, output_path, config_path, weights_path, classes_path, gpu)


