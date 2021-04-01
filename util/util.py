import numpy as np


def getRandomRGBColors(colorsAmount=100):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(colorsAmount, 3), dtype="uint8")


def readClasses(inputPath):
    with open(inputPath) as file:
        classNames = file.readlines()
    return [c.strip() for c in classNames]


def scale_abs(x, m=255):
    x = np.absolute(x)
    x = np.uint8(m * x / np.max(x))
    return x


def roi(gray, mn=125, mx=1200):
    m = np.copy(gray) + 1
    m[:, :mn] = 0
    m[:, mx:] = 0
    return m