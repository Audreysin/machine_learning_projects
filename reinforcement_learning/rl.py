import statistics
import math
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

FREE = 0
WALL = 1
PRIZE = 2
ALPHA_0 = 0.9
ALPHA_FINAL = 0.01
DISCOUNT_FACTOR = 0.95
TERMINATION_PROBABILITY = 1 - DISCOUNT_FACTOR

EMPTY_GRID = [
    # a    b    c    d    e
    [FREE, FREE, FREE, FREE, FREE],  # row 1
    [FREE, FREE, FREE, FREE, FREE],  # row 2
    [FREE, FREE, FREE, FREE, FREE],  # row 3
    [FREE, FREE, FREE, FREE, FREE],  # row 4
    [FREE, FREE, FREE, FREE, FREE]   # row 5
]

MAZE_GRID = [
    # a    b    c    d    e
    [FREE, FREE, FREE, FREE, FREE],  # row 1
    [FREE, WALL, WALL, WALL, FREE],  # row 2
    [FREE, FREE, FREE, WALL, FREE],  # row 3
    [FREE, WALL, FREE, FREE, FREE],  # row 4
    [FREE, WALL, WALL, WALL, WALL]  # row 5
]

FOUR_ROOM = [
    # a    b    c    d    e
    [FREE, FREE, WALL, FREE, FREE],  # row 1
    [FREE, FREE, FREE, FREE, FREE],  # row 2
    [WALL, WALL, FREE, WALL, WALL],  # row 3
    [FREE, FREE, FREE, FREE, FREE],  # row 4
    [FREE, FREE, WALL, FREE, FREE]   # row 5
]

UP = "up"
DOWN = "down"
RIGHT = "right"
LEFT = "left"

SENDER_PRIZE_ROW = "prize_row"
SENDER_PRIZE_COL = "prize_col"

RECEIVER_MESSAGE = "msg"
RECEIVER_ROW = "row_pos"
RECEIVER_COL = "col_pos"

MAX_Q_VAL = "max_q_val"
MAX_VAL_ACTION = "max_val_action"


class SenderState:
    def __init__(self, r_pos, c_pos):
        self.prize_row_pos = r_pos
        self.prize_col_pos = c_pos


class ReceiverState:
    def __init__(self, message, row_pos, col_pos):
        self.message = message
        self.row_pos = row_pos
        self.col_pos = col_pos


def sender_q_fn_init(n, grid):
    actions = [str(symbol) for symbol in range(n)]
    q_fn_rows = []
    for r in range(5):
        for c in range(5):
            if grid[r][c] == FREE and not(r == 2 and c == 2):
                q_fn_rows.append([r, c] + [0 for msg in actions])
    q_fn = pd.DataFrame(q_fn_rows, columns=([SENDER_PRIZE_ROW, SENDER_PRIZE_COL] + actions))
    return q_fn


def receiver_q_fn_init(n, grid):
    msg = [str(symbol) for symbol in range(n)]
    q_fn_rows = []
    for m in msg:
        for r in range(5):
            for c in range(5):
                if grid[r][c] == FREE:
                    q_fn_rows.append((m, r, c, 0, 0, 0, 0))
    q_fn = pd.DataFrame(q_fn_rows, columns=[RECEIVER_MESSAGE, RECEIVER_ROW, RECEIVER_COL, UP, DOWN, LEFT, RIGHT])
    return q_fn


def generate_prize_positions(grid):
    choices = []
    for r in range(5):
        for c in range(5):
            if grid[r][c] == FREE and not (r == 2 and c == 2):
                choices.append(f"{r} {c}")
    return choices


def generate_prize_position(possible_positions):
    pos = random.choice(possible_positions).split()
    return int(pos[0]), int(pos[1])


def decide_sender_next_action(epsilon, cur_state: SenderState, q_fn, action_set):
    take_random_choice = random.choices([True, False], weights=[epsilon, 1 - epsilon])[0]
    if take_random_choice:
        return random.choice(action_set)
    else:
        choices_df = q_fn[
            (q_fn[SENDER_PRIZE_ROW] == cur_state.prize_row_pos) &
            (q_fn[SENDER_PRIZE_COL] == cur_state.prize_col_pos)
        ]
        max_expected_reward = choices_df[action_set].max(axis=1).values[0]
        choices = choices_df.to_dict('records')[0]
        for action in action_set:
            if choices[action] == max_expected_reward:
                return action


def decide_receiver_next_action(epsilon, cur_state: ReceiverState, q_fn, action_set):
    take_random_choice = random.choices([True, False], weights=[epsilon, 1 - epsilon])[0]
    if take_random_choice:
        return random.choice(action_set)
    else:
        choices_df = q_fn[
            (q_fn[RECEIVER_MESSAGE] == cur_state.message) &
            (q_fn[RECEIVER_ROW] == cur_state.row_pos) &
            (q_fn[RECEIVER_COL] == cur_state.col_pos)
        ]
        max_expected_reward = choices_df[action_set].max(axis=1).values[0]
        choices = choices_df.to_dict('records')[0]
        for action in action_set:
            if choices[action] == max_expected_reward:
                return action


def get_new_position(move_dir, state: ReceiverState, grid):
    row_position = state.row_pos
    col_position = state.col_pos
    if move_dir == UP:
        row_position -= 1
    elif move_dir == DOWN:
        row_position += 1
    elif move_dir == RIGHT:
        col_position += 1
    else:
        col_position -= 1

    valid_move = True
    if row_position >= 5 or row_position <= -1:
        valid_move = False
    elif col_position >= 5 or col_position <= -1:
        valid_move = False
    elif grid[row_position][col_position] == WALL:
        valid_move = False

    return valid_move, row_position, col_position


def terminate():
    return random.choices([True, False], weights=[TERMINATION_PROBABILITY, 1 - TERMINATION_PROBABILITY])[0]


def take_random_action(epsilon):
    return random.choices([True, False], weights=[epsilon, 1 - epsilon])[0]


def q_learning_training(n_ep, n, epsilon, grid):
    # print("training")
    # s_time = time.time()
    sender_q_fn = sender_q_fn_init(n, grid)
    receiver_q_fn = receiver_q_fn_init(n, grid)
    sender_action_set = [str(x) for x in range(n)]
    receiver_action_set = [UP, DOWN, RIGHT, LEFT]
    alpha = ALPHA_0
    alpha_decrementor = (ALPHA_0 - ALPHA_FINAL) / (n_ep - 1)
    possible_prize_positions = generate_prize_positions(grid)
    # print("--- %s seconds init ---" % (time.time() - s_time))
    for episode in range(n_ep):
        # start_time = time.time()
        print(episode)
        prize_r_pos, prize_c_pos = generate_prize_position(possible_prize_positions)
        sender_cur_state = SenderState(prize_r_pos, prize_c_pos)
        # print(f"Prize at r: {prize_r_pos}, c: {prize_c_pos}")
        sender_q_fn[MAX_Q_VAL] = sender_q_fn[sender_action_set].max(axis=1)
        sender_q_fn[MAX_VAL_ACTION] = sender_q_fn[sender_action_set].idxmax(axis=1)

        sender_state_q_values = sender_q_fn[
            (sender_q_fn[SENDER_PRIZE_ROW] == sender_cur_state.prize_row_pos) &
            (sender_q_fn[SENDER_PRIZE_COL] == sender_cur_state.prize_col_pos)
        ]
        # print(sender_state_q_values)

        if sender_state_q_values[MAX_Q_VAL].values[0] == 0 or take_random_action(epsilon):
            sender_action_taken = random.choice(sender_action_set)
        else:
            sender_action_taken = sender_state_q_values[MAX_VAL_ACTION].values[0]
        # sender_action_taken = decide_sender_next_action(epsilon, sender_cur_state, sender_q_fn, sender_action_set)
        reward = 0.0
        # print(f"Sender action: {sender_action_taken}")
        receiver_cur_state = ReceiverState(sender_action_taken, 2, 2)
        # print("--- %s seconds pt1 ---" % (time.time() - start_time))
        while reward == 0 and not(terminate()):
            # start_time = time.time()
            # Receiver
            receiver_q_fn[MAX_Q_VAL] = receiver_q_fn[receiver_action_set].max(axis=1)
            receiver_q_fn[MAX_VAL_ACTION] = receiver_q_fn[receiver_action_set].idxmax(axis=1)

            receiver_state_q_values = receiver_q_fn[
                (receiver_q_fn[RECEIVER_MESSAGE] == receiver_cur_state.message) &
                (receiver_q_fn[RECEIVER_ROW] == receiver_cur_state.row_pos) &
                (receiver_q_fn[RECEIVER_COL] == receiver_cur_state.col_pos)
            ]

            if receiver_state_q_values[MAX_Q_VAL].values[0] == 0 or take_random_action(epsilon):
                receiver_action = random.choice(receiver_action_set)
                # If there is a tie, MAX_VAL_ACTION column will contain the first action it sees that tie
                # Choosing randomly prevents the agent from always choosing the same action when the the
                # q values for this action is zero
            else:
                receiver_action = receiver_state_q_values[MAX_VAL_ACTION].values[0]
            # receiver_action = decide_receiver_next_action(
            #     epsilon, receiver_cur_state, receiver_q_fn, receiver_action_set
            # )
            # print(receiver_state_q_values)
            # print(f"Receiver action {receiver_action}")
            valid_move, new_r, new_c = get_new_position(receiver_action, receiver_cur_state, grid)
            # print(f"New position {new_r}, {new_c} : {valid_move}")

            if (sender_cur_state.prize_row_pos == new_r
                    and sender_cur_state.prize_col_pos == new_c):
                reward = 1.0
                # print("Reward")

            receiver_q_value = receiver_state_q_values[receiver_action].values[0]
            if valid_move:
                next_state_max_q_value = receiver_q_fn[
                    (receiver_q_fn[RECEIVER_MESSAGE] == receiver_cur_state.message) &
                    (receiver_q_fn[RECEIVER_ROW] == new_r) &
                    (receiver_q_fn[RECEIVER_COL] == new_c)
                ][MAX_Q_VAL].values[0]
            else:
                next_state_max_q_value = receiver_state_q_values[MAX_Q_VAL].values[0]

            receiver_q_fn.loc[
                (receiver_q_fn[RECEIVER_MESSAGE] == receiver_cur_state.message) &
                (receiver_q_fn[RECEIVER_ROW] == receiver_cur_state.row_pos) &
                (receiver_q_fn[RECEIVER_COL] == receiver_cur_state.col_pos),
                receiver_action
            ] = (1 - alpha) * receiver_q_value + alpha * (reward + DISCOUNT_FACTOR * next_state_max_q_value)

            if valid_move:
                receiver_cur_state.row_pos = new_r
                receiver_cur_state.col_pos = new_c
            # print("--- %s seconds loop ---" % (time.time() - start_time))
            # if reward == 1 or terminate():
            #     break

        q_value = sender_state_q_values[sender_action_taken].values[0]
        sender_q_fn.loc[
            (sender_q_fn[SENDER_PRIZE_ROW] == sender_cur_state.prize_row_pos) &
            (sender_q_fn[SENDER_PRIZE_COL] == sender_cur_state.prize_col_pos),
            sender_action_taken
        ] = (1 - alpha) * q_value + alpha * reward
        alpha -= alpha_decrementor
    return sender_q_fn, receiver_q_fn


def q_learning_test(sender_q_fn, receiver_q_fn, grid, n):
    # print("Testing")
    # epsilon = 0
    total_discounted_reward = 0
    sender_action_set = [str(x) for x in range(n)]
    receiver_action_set = [UP, DOWN, RIGHT, LEFT]
    sender_q_fn[MAX_Q_VAL] = sender_q_fn[sender_action_set].max(axis=1)
    receiver_q_fn[MAX_Q_VAL] = receiver_q_fn[receiver_action_set].max(axis=1)
    sender_q_fn[MAX_VAL_ACTION] = sender_q_fn[sender_action_set].idxmax(axis=1)
    receiver_q_fn[MAX_VAL_ACTION] = receiver_q_fn[receiver_action_set].idxmax(axis=1)
    possible_prize_positions = generate_prize_positions(grid)

    for episode in range(1000):
        prize_r_pos, prize_c_pos = generate_prize_position(possible_prize_positions)
        sender_cur_state = SenderState(prize_r_pos, prize_c_pos)
        # sender_action_taken = decide_sender_next_action(epsilon, sender_cur_state, sender_q_fn, sender_action_set)
        sender_choices = sender_q_fn[
            (sender_q_fn[SENDER_PRIZE_ROW] == sender_cur_state.prize_row_pos) &
            (sender_q_fn[SENDER_PRIZE_COL] == sender_cur_state.prize_col_pos)
            ]
        if sender_choices[MAX_Q_VAL].values[0] == 0:
            sender_action_taken = random.choice(sender_action_set)
        else:
            sender_action_taken = sender_choices[MAX_VAL_ACTION].values[0]
        # sender_action_taken = sender_q_fn[
        #     (sender_q_fn[SENDER_PRIZE_ROW] == sender_cur_state.prize_row_pos) &
        #     (sender_q_fn[SENDER_PRIZE_COL] == sender_cur_state.prize_col_pos)
        # ][MAX_VAL_ACTION].values[0]
        receiver_cur_state = ReceiverState(sender_action_taken, 2, 2)
        reward = 0
        receiver_step_count = 0

        while reward == 0 and not(terminate()):
            # Receiver
            receiver_step_count += 1
            # receiver_action = decide_receiver_next_action(
            #     epsilon, receiver_cur_state, receiver_q_fn, receiver_action_set
            # )
            # receiver_action = receiver_q_fn[
            #     (receiver_q_fn[RECEIVER_MESSAGE] == receiver_cur_state.message) &
            #     (receiver_q_fn[RECEIVER_ROW] == receiver_cur_state.row_pos) &
            #     (receiver_q_fn[RECEIVER_COL] == receiver_cur_state.col_pos)
            # ][MAX_VAL_ACTION].values[0]
            receiver_choices = receiver_q_fn[
                (receiver_q_fn[RECEIVER_MESSAGE] == receiver_cur_state.message) &
                (receiver_q_fn[RECEIVER_ROW] == receiver_cur_state.row_pos) &
                (receiver_q_fn[RECEIVER_COL] == receiver_cur_state.col_pos)
                ]
            if receiver_choices[MAX_Q_VAL].values[0] == 0:
                receiver_action = random.choice(receiver_action_set)
                # If there is a tie, MAX_VAL_ACTION column will contain the first action it sees that tie
                # Choosing randomly prevents the agent from always choosing the same action when the the
                # q values for this action is zero
            else:
                receiver_action = receiver_choices[MAX_VAL_ACTION].values[0]

            valid_move, new_r, new_c = get_new_position(receiver_action, receiver_cur_state, grid)

            if (sender_cur_state.prize_row_pos == new_r
                    and sender_cur_state.prize_col_pos == new_c):
                reward = 1

            if valid_move:
                receiver_cur_state.row_pos = new_r
                receiver_cur_state.col_pos = new_c

            # if reward == 1 or terminate():
            #     break

        total_discounted_reward += (reward * (DISCOUNT_FACTOR**receiver_step_count))

    return total_discounted_reward / 1000


def generate_graph(df, epsilons, qpart):
    epsilon_colour_map = {
        0.01: 'red',
        0.1: 'blue',
        0.4: 'green'
    }
    for epsilon in epsilons:
        df_epsilon = df[df['epsilon'] == epsilon]
        plt.plot(df_epsilon["n_ep"], df_epsilon["avg_discounted_reward"], color=epsilon_colour_map[epsilon], label=f"Epsilon = {epsilon}")
        plt.errorbar(
            df_epsilon["n_ep"],
            df_epsilon["avg_discounted_reward"],
            yerr=df_epsilon["stdv"],
            fmt='o',
            ecolor=epsilon_colour_map[epsilon],
            color='black')
    plt.xlabel('log_base10(N_ep)')
    plt.ylabel('Average discounted reward')
    plt.title(f"Average discounted reward graph")
    plt.legend()
    # plt.show()
    plt.savefig(f"graph_part{qpart}.png")


def generate_graph_part_d(df, n_values, qpart):
    n_values_colour_map = {
        1: 'blue',
        2: 'red',
        4: 'blue',
        10: 'green',
        3: 'blue',
        5: 'green',
    }
    for n in n_values:
        df_epsilon = df[df['epsilon'] == n]
        plt.plot(df_epsilon["n_ep"], df_epsilon["avg_discounted_reward"], color=n_values_colour_map[n], label=f"N = {n}")
        plt.errorbar(
            df_epsilon["n_ep"],
            df_epsilon["avg_discounted_reward"],
            yerr=df_epsilon["stdv"],
            fmt='o',
            ecolor=n_values_colour_map[n],
            color='black')
    plt.xlabel('log_base10(N_ep)')
    plt.ylabel('Average discounted reward')
    plt.title(f"Average discounted reward graph")
    plt.legend()
    # plt.show()
    plt.savefig(f"graph_part{qpart}.png")


def part_c_tests():
    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    epsilons = [0.01, 0.1, 0.4]
    report_tuples = []
    for epsilon in epsilons:
        for n_ep in n_eps:
            avg_discounted_rewards_list = []
            for test in range(10):
                print(f"Test: {test} for epsilon = {epsilon}, episodes= {n_ep}")
                sender_q_fn, receiver_q_fn = q_learning_training(n_ep, 4, epsilon, FOUR_ROOM)
                avg_discounted_reward = q_learning_test(sender_q_fn, receiver_q_fn, FOUR_ROOM, 4)
                avg_discounted_rewards_list.append(avg_discounted_reward)
            report_tuples.append((
                epsilon,
                math.log10(n_ep),
                statistics.mean(avg_discounted_rewards_list),
                statistics.stdev(avg_discounted_rewards_list)
            ))

    performance_table = pd.DataFrame(report_tuples, columns=['epsilon', 'n_ep', 'avg_discounted_reward', 'stdv'])
    generate_graph(performance_table, epsilons, "c")


def part_d_tests():
    epsilon = 0.1
    n_values = [2, 4, 10]
    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    report_tuples = []
    for n in n_values:
        for n_ep in n_eps:
            avg_discounted_rewards_list = []
            for test in range(10):
                print(f"Test: {test} for N = {n}, episodes= {n_ep}")
                sender_q_fn, receiver_q_fn = q_learning_training(n_ep, n, epsilon, FOUR_ROOM)
                avg_discounted_reward = q_learning_test(sender_q_fn, receiver_q_fn, FOUR_ROOM, n)
                avg_discounted_rewards_list.append(avg_discounted_reward)
            report_tuples.append((
                n,
                math.log10(n_ep),
                statistics.mean(avg_discounted_rewards_list),
                statistics.stdev(avg_discounted_rewards_list)
            ))

    performance_table = pd.DataFrame(report_tuples, columns=['n', 'n_ep', 'avg_discounted_reward', 'stdv'])
    generate_graph_part_d(performance_table, n_values, "d")


def part_e_tests():
    epsilon = 0.1
    n_values = [2, 3, 5]
    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    report_tuples = []
    for n in n_values:
        for n_ep in n_eps:
            avg_discounted_rewards_list = []
            for test in range(10):
                print(f"Test: {test} for N = {n}, episodes= {n_ep}")
                sender_q_fn, receiver_q_fn = q_learning_training(n_ep, n, epsilon, MAZE_GRID)
                avg_discounted_reward = q_learning_test(sender_q_fn, receiver_q_fn, MAZE_GRID, n)
                avg_discounted_rewards_list.append(avg_discounted_reward)
            report_tuples.append((
                n,
                math.log10(n_ep),
                statistics.mean(avg_discounted_rewards_list),
                statistics.stdev(avg_discounted_rewards_list)
            ))

    performance_table = pd.DataFrame(report_tuples, columns=['n', 'n_ep', 'avg_discounted_reward', 'stdv'])
    generate_graph_part_d(performance_table, n_values, "f")


def part_f_tests():
    epsilon = 0.1
    n_values = [1]
    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    report_tuples = []
    for n in n_values:
        for n_ep in n_eps:
            avg_discounted_rewards_list = []
            for test in range(10):
                print(f"Test: {test} for N = {n}, episodes= {n_ep}")
                sender_q_fn, receiver_q_fn = q_learning_training(n_ep, n, epsilon, EMPTY_GRID)
                avg_discounted_reward = q_learning_test(sender_q_fn, receiver_q_fn, EMPTY_GRID, n)
                avg_discounted_rewards_list.append(avg_discounted_reward)
            report_tuples.append((
                n,
                math.log10(n_ep),
                statistics.mean(avg_discounted_rewards_list),
                statistics.stdev(avg_discounted_rewards_list)
            ))

    performance_table = pd.DataFrame(report_tuples, columns=['n', 'n_ep', 'avg_discounted_reward', 'stdv'])
    generate_graph_part_d(performance_table, n_values, "f")


def main():
    # part_c_tests()
    # part_d_tests()
    # part_e_tests()
    # part_f_tests()
    q_learning_training(100000, 4, 0.4, FOUR_ROOM)


if __name__ == "__main__":
    main()

