import sys
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.utils import to_categorical

from PIL import Image

from othello_ml import Othello, Action
from othello_ml.visualizer import Visualizer


class MLRenderer:
    def __init__(self, path='./result-prob'):
        self.count = 0
        self.path = path

    def render(self, board):
        board = board.reshape((8, 8)) * 10000
        print(board)
        image = Image.fromarray(board.astype('uint8'), 'L')
        image = image.resize((400, 400))
        image.save(f"{self.path}task{self.count}.png")
        self.count += 1


class MLAgent:
    def __init__(self,
                 othello,
                 model,
                 eps=0.999,
                 random_rate=1.,
                 no_reward=False,
                 renderer=None):
        self.random_rate = random_rate
        self.othello = othello
        self.directions = self.othello.directions
        self.model = model
        self.eps = eps
        self.turn = None
        self.renderer = renderer
        self.sorted_prediction = None
        self.predict_before = False
        self.predict_index = 0

        othello.agent_actor(self.act)
        if not no_reward:
            othello.agent_reward(self.reward)

    def _is_opposites_nearby(self, board, point, opposite):
        if board[point[0]][point[1]] is not 0:
            return False
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
            if self.predict_before:
                self.predict_index += 1
                action.x, action.y = self.sorted_prediction[self.predict_index]
                return action, False

            places = self.get_available_places(board, my_disk)
            self.predict_before = False

            if places:
                action.x, action.y = places[random.randrange(len(places))]
                return action, False
            else:
                return None, True

        self.random_rate *= self.eps
        if self.random_rate > np.random.uniform():
            places = self.get_available_places(board, my_disk)
            self.predict_before = False

            if places:
                action.x, action.y = places[random.randrange(len(places))]
                return action, False
            else:
                return None, True

        self.turn = turn
        self.predict_index = 0

        board = [[1 if i is my_disk else 0 if i is 0 else -1 for i in sl]
                 for sl in board]
        result = self.model.predict(
            np.reshape(np.array([board]), (1, 8, 8, 1)))[:, :, :, 1]
        if self.renderer is not None:
            self.renderer.render(result)

        self.sorted_prediction = np.dstack(
            np.unravel_index(np.argsort(result.ravel()), (8, 8)))[0]
        action.x, action.y = self.sorted_prediction[0]
        self.predict_before = True
        return action, False

    def reward(self, boards, reward):
        my_disk = -1 if self.turn is 0 else 1

        converted = []
        actions = []

        for action, board in boards:
            converted.append([[(1 if i is my_disk else 0 if i is 0 else -1)
                               for i in sl] for sl in board])
            ac = np.zeros((8, 8))
            ac[(action.x, action.y)] = 1
            actions.append(ac)

        converted = np.array(converted)
        actions = np.array(actions)

        if reward > 0:
            model.fit(
                converted.reshape((-1, 8, 8, 1)),
                to_categorical(actions.reshape((-1, 8, 8, 1))),
                epochs=reward)


if __name__ == "__main__":

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

    model.compile(
        'adam', loss='categorical_crossentropy', metrics=['accuracy'])

    experience_replay = []
    num_of_episode = 1000

    result = model.predict(np.zeros((1, 8, 8, 1)))
    result.shape

    for i in range(num_of_episode):
        print(f"episode {i}")
        othello = Othello()
        agent1 = MLAgent(othello, model)
        agent2 = MLAgent(othello, model)
        # visualizer = Visualizer(othello, path=f'./result/{i}')

        othello.play()

    model.save_weights('episode_1000.h5')
