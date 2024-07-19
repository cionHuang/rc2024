import numpy as np
import pickle
import logging
from itertools import product
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
BOARD_ROWS = 3
BOARD_COLS = 5
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


def is_real(comb_arr):
    """
    判断生成的棋盘的每一列是否符合现实情况（即每一列空的位置不能在球的下方）。

    Args:
        comb_arr (np.ndarray): 一个二维数组，表示数独棋盘，其中每个元素为整数（1-9）或0（表示空位）。

    Returns:
        bool: 如果每一列都是合法的（即没有重复的数字，且都填满了数字），则返回True；否则返回False。

    """
    i = 0
    for col in range(BOARD_COLS):
        col_values = comb_arr[:, col]
        zero_positions = np.where(col_values == 0)[0].tolist()
        if zero_positions == [0, 1, 2] or zero_positions == [0, 1] or zero_positions == [0] or zero_positions == []:
            i += 1
    if i == BOARD_COLS:
        return True
    return False


class GetState:
    def __init__(self):
        self.all_states = {}
        self.generate()

    def generate(self):
        '''
        生成所有可能的状态，并保存在字典中
        '''
        print("Generating all_states...")
        self.get_all_states(self.all_states)
        with open('../models/all_states.pickle', 'wb') as f:
            pickle.dump(self.all_states, f)

    def get_all_states(self, all_states):
        """
        遍历所有可能的状态，从3^15个状态中,选出满足物理规律的状态，存入all_states字典中中。

        Args:
            self (object): 类实例对象。
            all_states (dict): 用于存储状态的字典，键为状态哈希值，值为一个包含三个元素的列表
                [当前状态对象, 是否为结束状态, 非零元素个数]。
        """
        all_combinations = product([-1, 0, 1], repeat=15)
        for combination in tqdm(all_combinations, total=3**15):
            current_state = State()
            comb_np = np.array(combination).reshape(3, 5)
            if is_real(comb_np):
                current_state.data = comb_np
                new_hash = current_state.hash()
                is_end = current_state.is_end()
                non_zero_count = np.count_nonzero(comb_np)
                all_states[new_hash] = [current_state, is_end, non_zero_count]


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

        #  检查是否平局（所有位置都被填满）
        if np.all(self.data != 0):
            self.winner = 0
            self.end = True
            return self.end

        # 游戏还未结束
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
            col = self.data[:, j]
            if np.any(np.all(col[:, None] == win_pattern, axis=0)):
                completed_cols += 1
                if completed_cols >= 3:
                    return True
        return False


if __name__ == '__main__':
    temp = GetState()
