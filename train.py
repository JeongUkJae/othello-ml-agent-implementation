import sys
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization

from othello_ml import Othello, Action


class MLAgent:
    def __init__(self, othello, model, eps=0.999):
        self.random_rate = 1.
        self.othello = othello
        self.directions = self.othello.directions
        self.model = model
        self.eps = eps
        self.turn = None

        othello.agent_actor(self.act)
        othello.agent_reward(self.reward)

    def _is_opposites_nearby(self, board, point, opposite):
        for dir in self.directions:
            x = point[0] + dir[0]
            y = point[1] + dir[1]

            if x < 0 or x > 7 or y < 0 or y > 7:
                continue

            if board[x][y] is opposite:
                return True

        return False

    def get_available_places(self, board, disk):
        places = []
        opposite = -1 if disk is 1 else 1

        for x in range(8):
            for y in range(8):
                if self._is_opposites_nearby(board, (x, y), opposite):
                    action = Action()
                    action.x, action.y = x, y

                    if self.othello._is_valid_action(0 if disk is -1 else 1,
                                                     action):
                        places.append((x, y))

        return places

    def act(self, board, turn, invalid_before):
        action = Action()
        my_disk = -1 if turn is 0 else 1
        if invalid_before:
            places = self.get_available_places(board, my_disk)

            action.x, action.y = places[random.randrange(len(places))]
            return action, False

        self.random_rate *= self.eps
        if self.random_rate > np.random.uniform():
            places = self.get_available_places(board, my_disk)

            action.x, action.y = places[random.randrange(len(places))]
            return action, False

        self.turn = turn

        board = [[1 if i is my_disk else 0 if i is 0 else -1 for i in sl]
                 for sl in board]
        result = self.model.predict(
            np.reshape(np.array([board]), (1, 8, 8, 1)))
        _, action.x, action.y = np.unravel_index(result.argmax(),
                                                 result.shape)[1:-1]
        return action, False

    def reward(self, boards, reward):
        my_disk = -1 if self.turn is 0 else 1

        converted = []
        actions = []

        for action, board in boards:
            converted.append([[(1 if i is my_disk else 0 if i is 0 else -1)
                               for i in sl] for sl in board])
            ac = np.zeros(8, 8)
            ac[(action.x, action.y)] = 1
            actions.append(ac)

        converted = np.array(converted)
        actions = np.array(actions)

        model.fit(
            converted.reshape((-1, 8, 8, 1)),
            actions.reshape((-1, 8, 8, 1)),
            epochs=10)


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
    Conv2D(1, (1, 1), activation='softmax', padding='same'),
])

model.summary()

model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

experience_replay = []
num_of_episode = 1000

result = model.predict(np.zeros((1, 8, 8, 1)))
result.shape

for i in range(num_of_episode):
    othello = Othello(verbose=True)
    agent1 = MLAgent(othello, model)
    agent2 = MLAgent(othello, model)

    othello.play()
