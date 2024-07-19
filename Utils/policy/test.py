import numpy as np
import pickle
import logging
logging.basicConfig(level=logging.INFO)
BOARD_ROWS = 3
BOARD_COLS = 5
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
all_states = {}
estimations = {}
STATES_PATH = '../models/all_states.pickle'
POLICY_PATH = '../models/policy.bin'


class State:
    def __init__(self):
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS)).astype(int)
        self.winner = None
        self.hash_val = None
        self.end = None

    def hash(self):
        """
        计算并返回对象的哈希值。
        """
        if self.hash_val is None:
            self.hash_val = 0
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
                if i == -1:
                    i = 2
                self.hash_val = self.hash_val * 3 + i
        return int(self.hash_val)

    def is_end(self):
        """
        判断游戏是否结束。

        Returns:
            bool: 如果游戏结束返回True，否则返回False。
        """
        if self.end is not None:
            return self.end

        red_win = self.check_win(1)
        blue_win = self.check_win(-1)

        if red_win:
            self.winner = 1
            self.end = True
            return self.end
        elif blue_win:
            self.winner = -1
            self.end = True
            return self.end

        if np.all(self.data != 0):
            self.winner = 0
            self.end = True
            return self.end

        self.end = False
        return self.end

    def check_win(self, p):
        '''
        返回当前状态下，标号为p的玩家已满足胜利条件的列数。
        '''
        win_pattern = np.array([[p, p, p],
                                [p, -p, p],
                                [p, p, -p]])
        completed_cols = 0

        for j in range(BOARD_COLS):
            column = self.data[:, j]
            if np.any(np.all(column[:, None] == win_pattern, axis=0)):
                completed_cols += 1
        if completed_cols == 3:
            return True
        return False

    def check_limit(self, j):
        '''
        检查第j列是否可以落子（即该列是否已经满了）。
        '''
        num_empty = np.sum(self.data[:, j] == 0)
        if 1 <= num_empty <= 3:
            return True

    def update_state(self, comb, symbol):
        '''
        更新状态，将棋盘上第j列最底部的空白格填充为symbol（1/-1）方符号。
        '''
        new_state = State()
        new_state.data = np.copy(self.data)
        for col in comb:
            for i in reversed(range(3)):
                if new_state.data[i, col] == 0:
                    new_state.data[i][col] = symbol
                    break
        return new_state

    def print(self):
        """
        打印棋盘，用于游戏测试。
        """
        for i in range(0, BOARD_ROWS):
            print('---------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '0'
                if self.data[i, j] == 0:
                    token = '·'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('---------------------')
        print('')


class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None

    def alternate(self):
        while True:
            yield self.p2
            yield self.p1

    def play(self):
        """
        执行游戏，并返回赢家。
        """
        alternator = self.alternate()
        current_state = State()

        while True:
            player = next(alternator)
            actions = player.act(current_state)
            if actions == ():
                continue
            current_state = current_state.update_state(actions, player.symbol)
            next_state_hash = current_state.hash()

            current_state, is_end, _ = all_states[next_state_hash]

            if is_end:
                return current_state.winner


class Player:
    def __init__(self, symbol):
        self.symbol = symbol

    def act(self, state):
        '''
        根据当前状态进行利用，返回选择价值最高的列。
        '''
        next_positions = []

        for j in range(BOARD_COLS):
            if state.check_limit(j):
                next_positions.append(j)
        values = []
        for j in next_positions:
            next_state = state.update_state((j,), self.symbol)
            next_state_hash = next_state.hash()
            values.append((estimations[next_state_hash], j))

        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        best_action = values[0][1]
        return (best_action,)


class HumanPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.keys = {'z': 0, 'x': 1, "c": 2, "v": 3, "b": 4}

    def act(self, state):
        '''
        获取用户输入的动作序列，并将其转换成合法的列索引。
        '''
        state.print()
        while True:
            try:
                col_inputs = input("Choose (0~5) from 'z' 'x' 'c' 'v 'b', split by space: ").split()
                actions = tuple(self.keys[col] for col in col_inputs)

                if any(j < 0 or j >= BOARD_COLS for j in actions):
                    logging.warning("Invalid column!")
                    continue

                if any(not state.check_limit(j) for j in actions):
                    logging.warning("This column is full!")
                    continue
                break
            except (ValueError, KeyError):
                logging.warning("Invalid input.")

        return actions


def load_states_and_policy():
    """
    从指定的路径加载所有状态信息和策略加载到全局变量中。
    """
    global all_states
    global estimations
    try:
        logging.info("Loading all_states...")
        with open(STATES_PATH, 'rb') as f:
            all_states = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Path: {STATES_PATH} not found")

    try:
        logging.info("Loading policy...")
        with open(POLICY_PATH, 'rb') as f:
            estimations = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Path: {POLICY_PATH} not found")


def test():
    '''
    主函数，用于启动游戏并进行交互操作。
    '''
    while True:
        ai = Player(1)
        human = HumanPlayer(-1)
        judger = Judger(ai, human)
        winner = judger.play()
        if winner == ai.symbol:
            print("You lose!")
        elif winner == human.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    load_states_and_policy()
    test()
