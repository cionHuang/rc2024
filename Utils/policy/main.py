import numpy as np
import pickle
import logging
logging.basicConfig(level=logging.INFO)

all_states = {}
estimations = {}


class State:
    def __init__(self):
        self.data = np.zeros((3, 5)).astype(int)
        self.winner = None
        self.hash_val = None
        self.end = None

    def hash(self):
        """
        计算并返回对象的哈希值。
        """
        if self.hash_val is None:
            self.hash_val = 0
            for i in self.data.reshape(15):
                if i == -1:
                    i = 2
                self.hash_val = self.hash_val * 3 + i
        return int(self.hash_val)

    def check_limit(self, j):
        '''
        检查第j列是否可以落子（即该列是否已经满了）。
        '''
        num_empty = np.sum(self.data[:, j] == 0)
        if 1 <= num_empty <= 3:
            return True

    def update_state(self, col):
        '''
        更新状态，将棋盘上第j列最底部的空白格填充为己方符号。
        '''
        new_state = State()
        new_state.data = np.copy(self.data)
        for i in reversed(range(3)):
            if new_state.data[i][col] == 0:
                new_state.data[i][col] = 1
                break
        return new_state


def load_states(STATES_PATH):
    global all_states
    logging.info("Loading all_states...")
    with open(STATES_PATH, 'rb') as f:
        all_states = pickle.load(f)
    logging.info("States Loaded.")


def load_policy(POLICY_PATH):
    global estimations
    logging.info(f"Loading policy from:{POLICY_PATH}...")
    with open(POLICY_PATH, 'rb') as f:
        estimations = pickle.load(f)
    logging.info("Policy Loaded.")


def select_column(lst):
    priority = [3, 4, 2, 5, 1]
    for number in priority:
        if number in lst:
            return number

    return None


def run_policy(board):
    """
    根据给定的棋盘状态执行策略，返回最优落子位置及其对应的价值估计列表。

    Args:
        board (List[int]): 长度为25的整数列表，表示五子棋棋盘的状态。
            其中0表示空位，1表示我方棋子，2表示对方棋子。

    Returns:
        tuple: 包含两个元素的元组。
        - int: 最优落子位置（从1开始编号）。
        - List[tuple]: 包含多个二元组的列表，每个二元组代表一个可能的落子位置及其对应的价值估计。
            每个二元组的第一个元素是价值估计（float类型），第二个元素是落子位置（int类型，从0开始编号）。

    """
    current_state = State()
    current_state.data = np.array(board)

    # 检查是否获胜
    if all_states[int(current_state.hash())][1] == True:
        print('已大胜，开始就近选择!')
        next_positions = []
        for j in range(5):
            if current_state.check_limit(j):
                next_positions.append(j + 1)

        column = select_column(next_positions)
        return column, 0

    # 如果没有获胜，则返回最优选择
    next_positions = []
    for j in range(5):
        if current_state.check_limit(j):
            next_positions.append(j)

    values = []
    for j in next_positions:
        next_state = current_state.update_state(j)
        next_state_hash = next_state.hash()
        values.append((estimations[next_state_hash], j))

    np.random.shuffle(values)
    values.sort(key=lambda x: x[0], reverse=True)
    best_action = values[0][1] + 1
    return best_action, values


if __name__ == '__main__':
    load_states('./models/all_states.pickle')
    load_policy('./models/policy.bin')

    board = [[-1, 0, -1, -1, -1],
             [1, 1, -1, -1, -1],
             [-1, 1, -1, -1, -1]]

    barn_ai, value = run_policy(board)
    print(barn_ai, value)
