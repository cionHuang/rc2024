import numpy as np
import pickle
import logging
import random
logging.basicConfig(level=logging.INFO)

BOARD_ROWS = 3
BOARD_COLS = 5
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
all_states = {}
estimations = {}
STATES_PATH = '../models/all_states.pickle'
SAVE_PATH = '../models/policy.bin'


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
        if self.hash_val is not None:
            return self.hash_val

        self.hash_val = 0
        for i in self.data.ravel():
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

    def update_state(self, j):
        '''
        更新状态，将棋盘上第j列最底部的空白格填充为己方符号。
        '''
        new_state = State()
        new_state.data = np.copy(self.data)
        lowest_i = -1
        for i, row in enumerate(self.data):
            if row[j] == 0:
                lowest_i = i
        new_state.data[lowest_i][j] = 1
        return new_state


class Player:
    def __init__(self, step_size, epsilon):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []

    def reset(self):
        '''
        重置玩家状态，包括清空状态列表和贪婪列表。
        '''
        self.states = []
        self.greedy = []

    def set_state(self, state):
        '''
        设置玩家状态，包括添加新状态到状态列表和贪婪列表。
        '''
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        '''
        设置玩家符号，并初始化估值字典。
        '''
        self.symbol = symbol
        for hash_val in all_states.keys():
            (state, is_end, _) = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    def backup(self):
        '''
        使用时序差分算法更新估值。
        '''
        self.states = [state.hash() for state in self.states]
        for i in reversed(range(len(self.states) - 1)):
            state = self.states[i]
            td_error = self.greedy[i] * (self.estimations[self.states[i + 1]] - self.estimations[state])
            self.estimations[state] += self.step_size * td_error

    def act(self):
        '''
        根据当前状态进行探索和利用，返回最优选择的列。
        '''
        # 取最新的状态
        state = self.states[-1]
        next_positions = []

        # 获取下一步可以落子的列
        for j in range(BOARD_COLS):
            if state.check_limit(j):
                next_positions.append(j)

        # 若选择探索，则随机选择一列，防止陷入局部最优解。（概率10%）
        if np.random.rand() < self.epsilon:
            action_col = np.random.choice(next_positions)
            self.greedy[-1] = False
            return action_col

        # 若选择利用，则返回价值最大的列
        values = []
        for j in next_positions:
            next_state = state.update_state(j)
            next_state_hash = next_state.hash()
            values.append((self.estimations[next_state_hash], j))

        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        best_action = values[0][1]
        return best_action

    def save_policy(self, path):
        '''
        保存策略至指定路径。
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.estimations, f)


class Judger:
    def __init__(self, player) -> None:
        self.player = player
        self.player.set_symbol(1)

    def self_play(self, state):
        '''
        贪心思想：从给定状态开始只有我方下棋，直至我方获胜。
        '''
        self.player.set_state(state)
        while True:
            column = self.player.act()
            state = state.update_state(column)
            next_hash = state.hash()
            state, is_end, _ = all_states[next_hash]
            self.player.set_state(state)
            if is_end:
                break


def load_states():
    """
    从指定的路径加载所有状态信息到全局变量all_states中。
    """
    global all_states
    try:
        logging.info("Loading all_states...")
        with open(STATES_PATH, 'rb') as f:
            all_states = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Path: {STATES_PATH} not found")


def train(epochs, step_size, epsilon):
    """
    训练函数，用于训练一个玩家，使其通过自我对弈来优化策略。

    Args:
        epochs (int): 训练的总轮数。
        step_size (float): 玩家在策略更新时的步长。
        epsilon (float): 探索率，表示玩家在决策时随机选择的概率。

    Returns:
        None

    """
    player1 = Player(step_size, epsilon)
    judger = Judger(player1)
    for i in range(epochs):
        keys = list(all_states.keys())
        random.shuffle(keys)
        for key in keys:
            state, is_end, n_balls = all_states[key]
            if not is_end:
                judger.self_play(state)
            player1.backup()
            player1.reset()
    player1.save_policy(SAVE_PATH)
    print(f'Saved policy!')


if __name__ == '__main__':
    load_states()
    epoch = 20
    step_size = 0.01
    epsilon = 0.2
    train(epoch, step_size, epsilon)
