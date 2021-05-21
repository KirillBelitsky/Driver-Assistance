import argparse
from types import SimpleNamespace

from train.trainNeuralNetwork import TrainNeuralNetwork

parser = argparse.ArgumentParser()

parser.add_argument('--weights', required=False, help='path to pretrained weights')
parser.add_argument('--tiny', required=False, help='is tiny model')

args = vars(parser.parse_args())

weights = args['weights'] if args['weights'] is not None else 'config/yolov3.weights'
tiny = args['tiny'] if args['tiny'] is not None else False

FLAGS = SimpleNamespace(weights=weights, tiny=tiny)

train = TrainNeuralNetwork()
train.train(FLAGS)

print('The learning is ended successfully!')