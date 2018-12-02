import sys
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.utils import to_categorical

from othello_ml import Othello, Action
from othello_ml.visualizer import Visualizer

from train import MLAgent, MLRenderer


class CliAgent:
    def __init__(self, othello):
        self.othello = othello
        othello.agent_actor(self.act)

    def act(self, board, turn, invalid_before):
        if invalid_before:
            print("정상적인 수를 두시기 바랍니다.")

        for row in board:
            print(
                "|", "|".join(
                    map(lambda x: 'y' if x is 1 else 'n' if x is -1 else 'O',
                        row)), "|")

        is_pass = 1 if input('패스입니까? yn') == 'y' else 0
        try:
            x = int(input('x:'))
            y = int(input('y:'))
        except:
            x = y = 0
            is_pass = True
            print('제대로 된 입력이 아니기 때문에 패스합니다.')

        action = Action(x=x, y=y)
        return action, is_pass


model = Sequential([
    Conv2D(
        32,
        kernel_size=(3, 3),
        input_shape=(8, 8, 1),
        padding='same',
        activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(2, (1, 1), activation='softmax', padding='same'),
])

model.summary()

model.load_weights("episode_1000.h5")

while True:
    othello = Othello()
    renderer = MLRenderer(path='./result/test-prob-')
    agent1 = MLAgent(
        othello, model, random_rate=0, no_reward=True, renderer=renderer)
    agent2 = CliAgent(othello)
    visualizer = Visualizer(othello, path=f'./result/test-')

    othello.play()
