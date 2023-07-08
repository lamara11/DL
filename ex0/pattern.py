import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        if resolution % 2*tile_size != 0:
            return
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution))
        zeros=np.zeros((self.tile_size,self.tile_size))
        ones=np.ones((self.tile_size,self.tile_size))
        total= np.concatenate((np.concatenate((zeros,ones),axis=0), np.concatenate((ones,zeros),axis=0)),axis=1)
        print("total")
        print(np.shape(total))
        print(total)
        self.output=np.tile(total, (self.resolution//(2*self.tile_size), self.resolution//(2*self.tile_size)))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        dist = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        self.output = np.zeros((self.resolution, self.resolution))
        self.output[dist <= self.radius] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution, 3))

    def draw(self):
        x=np.linspace(0, 1, self.resolution)
        blue, red = np.meshgrid(x, x)
        g=1-(blue)
        self.output[:, :, 0] = blue
        self.output[:, :, 1] = red
        self.output[:, :, 2] = g
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()