import argparse
import os
from detector import Detector

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='path to input video')
parser.add_argument('--output', required=True, help='path to output video')
parser.add_argument('--config', required=True, help='path to yolo config files')
parser.add_argument('--gpu', required=False, help='GPU Cuda must be used (0 - no, 1 - yes')

args = vars(parser.parse_args())

inputPath = args['input']
outputPath = args['output']
gpu = args['gpu']
gpu = True if gpu is not None and gpu == '1' else False

weightsPath = os.path.sep.join([args["config"], "yolov4-tiny.weights"])
configPath = os.path.sep.join([args["config"], "yolov4-tiny.cfg"])

detector = Detector()
detector.detect(inputPath, outputPath, configPath, weightsPath, gpu)


