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

        self.image = np.zeros((401, 401, 3))
        self.image[:, :] = np.array([128, 128, 128])

        for i in range(9):
            k = i * 50
            self.image[k, ] = self.image[:, k] = np.array([255, 255, 255])

    def render(self, board):
        for i in range(8):
            for j in range(8):
                self.image[i * 50 + 1:i * 50 +
                           50, j * 50 + 1:j * 50 + 50] = np.array([
                               128, 128, 128
                           ]) if board[i][j] is 0 else np.array([
                               255, 255, 255
                           ]) if board[i][j] is -1 else np.array([0, 0, 0])
        image = Image.fromarray(self.image.astype('uint8'))
        image.save(f"{self.path}task{self.count}.png")
        self.count += 1
