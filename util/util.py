import numpy as np


def getRandomRGDColors(colorsAmount=100):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(colorsAmount, 3), dtype="uint8")
