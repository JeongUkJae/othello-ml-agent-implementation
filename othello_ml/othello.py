from itertools import permutations

from functools import wraps


class Action:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"<Action x={self.x} y={self.y}>"


class Othello:
    """Othello Game
    """

    def __init__(self, verbose=False):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.board[3][4] = self.board[4][3] = 1
        self.board[3][3] = self.board[4][4] = -1

        self.is_end_of_game = False
        self.agents = []
        self.agents_reward = []
        self.renderers = []
        self.agents_actions = [[], []]

        self.directions = [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, 1),
                           (1, 0), (1, -1)]
        self.verbose = verbose

    def _is_valid_action(self, turn, action):
        """validate action is valid
        """
        if not isinstance(action, Action):
            raise TypeError("Action must be Action class")

        if action.x < 0 or action.x > 7 or action.y < 0 or action.y > 7:
            raise Exception("You must set disk in board.")

        if self.board[action.x][action.y] is not 0:
            return False

        agent = -1 if turn is 0 else 1

        for direction in self.directions:
            for i in range(1, 9):
                x = action.x + i * direction[0]
                y = action.y + i * direction[1]

                if x < 0 or x > 7 or y < 0 or y > 7:
                    break

                if self.board[x][y] is 0:
                    break
                elif self.board[x][y] is agent:
                    # if meet agent's disk and no opposite's dist in between, not valid
                    if i is not 1:
                        return True

                    break
                else:
                    continue

        # what is wrong?
        return False

    def _get_valid_directions(self, turn, action):
        """get valid direcitons by action
        """
        if not isinstance(action, Action):
            raise TypeError("Action must be Action class")

        if action.x < 0 or action.x > 7 or action.y < 0 or action.y > 7:
            raise Exception("You must set disk in board.")

        if self.board[action.x][action.y] is not 0:
            return []

        agent = -1 if turn is 0 else 1
        dirs = []

        for direction in self.directions:
            for i in range(1, 9):
                x = action.x + i * direction[0]
                y = action.y + i * direction[1]

                if x < 0 or x > 7 or y < 0 or y > 7:
                    break

                if self.board[x][y] is 0:
                    break
                elif self.board[x][y] is agent:
                    # if meet agent's disk and no opposite's dist in between, not valid
                    if i is 1:
                        break

                    dirs.append(direction)
                    break
                else:
                    continue

        return dirs

    def _check_end_of_game(self, force=False):
        """check no empty place on board
        """
        if not force:
            self.is_end_of_game = not bool(
                sum(sum(x is 0 for x in sl) for sl in self.board))
        else:
            self.is_end_of_game = True
        flatten = sum(self.board, [])
        self.score = 32 - sum(1 for item in flatten if item is 1)

    def _act(self, turn, action, directions):
        disk = -1 if turn is 0 else 1
        self.board[action.x][action.y] = disk

        for direction in directions:
            for i in range(1, 9):
                x = action.x + i * direction[0]
                y = action.y + i * direction[1]

                if self.board[x][y] is disk:
                    break

                self.board[x][y] = disk

    def agent_actor(self, func):
        """register function as agent actor
        """
        if len(self.agents) is 2:
            raise Exception("You cannot register 3 or more agents.")

        self.agents.append(func)
        return func

    def agent_reward(self, func):
        """register function as agent actor
        """
        if len(self.agents_reward) is 2:
            raise Exception("You cannot register 3 or more agents_reward.")

        self.agents_reward.append(func)
        return func

    def renderer(self, func):
        """regiseter function as renderer
        """
        self.renderers.append(func)
        return func

    def play(self):
        """play othello
        """
        if len(self.agents) is not 2:
            raise Exception("You have to register 2 agents to play Othello.")

        turn = 0
        invalid_before = False
        pass_both = 0

        while not self.is_end_of_game:
            action, is_pass = self.agents[turn](self.board, turn,
                                                invalid_before)

            if self.verbose:
                print(turn, action)

            if is_pass:
                turn = int(not turn)
                pass_both += 1

                if pass_both == 2:
                    self._check_end_of_game(force=True)
                continue

            dirs = self._get_valid_directions(turn, action)

            if dirs:
                self.agents_actions[turn].append((action, self.board[:][:]))
                self._act(turn, action, dirs)

                for rd in self.renderers:
                    rd(self.board)
            else:
                invalid_before = True
                continue

            # toggle turn
            pass_both = 0
            turn = int(not turn)
            invalid_before = False
            self._check_end_of_game()

        for i in range(len(self.agents_reward)):
            self.agents_reward[i](self.agents_actions[i], -self.score)
