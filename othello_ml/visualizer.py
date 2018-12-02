import numpy as np

from PIL import Image, ImageTk

from .othello import Othello


class Visualizer:
    def __init__(self, othello, path='./result'):
        if not isinstance(othello, Othello):
            raise Exception("Object othello must be an instance of Othello.")
        self.othello = othello
        self.othello.renderer(self.render)
        self.count = 0
        self.path = path

        self.image = np.zeros((401, 401))
        self.image[:, :] = 128

        for i in range(9):
            k = i * 50
            self.image[k, ] = self.image[:, k] = 255

    def render(self, action, turn):
        self.image[action.y * 50 + 1:action.y * 50 + 50, action.x * 50 +
                   1:action.x * 50 + 50] = turn * 255
        image = Image.fromarray(self.image.astype('uint8'), 'L')
        image.save(f"{self.path}/task{self.count}.png")
        self.count += 1
