import numpy as np
import matplotlib.pyplot as plt
from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def main():
    label_path = './Labels.json'
    file_path = './exercise_data/'
    gen=ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False,
                             shuffle=True)
    gen.next()
    gen.show()

    checkerboard = Checker(resolution=32, tile_size=8)
    checkerboard.draw()
    checkerboard.show()

    circle = Circle(resolution=512, radius=100, position=(256, 256))
    circle.draw()
    circle.show()

    spectrum = Spectrum(resolution=32)
    spectrum.draw()
    spectrum.show()


if __name__ == '__main__':
    main()