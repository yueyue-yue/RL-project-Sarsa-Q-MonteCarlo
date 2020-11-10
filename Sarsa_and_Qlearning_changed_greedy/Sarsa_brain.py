import numpy as np
import pandas as pd
import time
import random
class SARSA:
    def __init__(self, greedy_rate, learning_rate, reward_decay):
        self.greedy = greedy_rate
        self.actions = [0, 1, 2, 3]
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    #  Creat the data frame of table value like:
    #       actions  0            1          2            3  ( 上 下 左 右 )/(up down left right）
    #  state
    #  [x,y]         q([x,y],0)   q([x,y],1)  q([x,y],2)   q([x,y],3)
    # The location of the agent is viewed as a signal of the state.

    def check_state_exist(self, state):
        #  If this is a new [x,y] state, we append the element to the table.
        #  The element is like [x,y] : 0,0,0,0
        if state not in self.table.index:
            # append new state to q table
            self.table = self.table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.table.columns,
                    name=state,
                )
            )

    def change_greedy(self, greedy):
        self.greedy = greedy

    def table_calculate_(self, state_current_, action_current_, state_next, action_next, reward_current,
                         step_info):

        current_value = self.locate_table_value(state_current_, action_current_)

        next_value = self.locate_table_value(state_next, action_next)

        # print(f'现在状态:{state_current_} 现在行为:{action_current_} 现在的值：{current_value}')
        # print(f'下一状态:{state_next} 下一的行为：{action_next}')
        step_size = 1 / step_info
        renew_value = current_value + step_size * (reward_current + self.learning_rate * next_value - current_value)
        self.renew_table(state_current_, action_current_, renew_value)
        return 0

    def renew_table(self, location, action_renew, reward_renew):
        # renew the q([x,y],a) value to the table
        location = str(location)
        self.table.loc[location, action_renew] = reward_renew
        return 0

    def locate_table_value(self, location, a):
        location = str(location)
        return self.table.loc[location, a]

    def choose_action(self, location):
        # action selection
        def chose_again(value):
            action = value
            while action == value:
                action = int(np.random.uniform(0, 4))
            return action

        state_old = str(location)
        if np.random.rand() < self.greedy:
            # choose best action
            # some actions may have the same value, randomly choose on in these actions
            # action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            # find best policy based on the current table
            #
            action_list = []
            key_value = {}
            key_value[0] = self.locate_table_value(state_old, 0)
            key_value[1] = self.locate_table_value(state_old, 1)
            key_value[2] = self.locate_table_value(state_old, 2)
            key_value[3] = self.locate_table_value(state_old, 3)
            # This is to take the situation that many equal values in the (q,a) table.
            # In this case, we first reverse the dict of key_value in sequence
            # Chose element from the back of the dict (max value), and append those the same as the biggest value
            # into the action_list, then randomly chose the action.
            # Base on this method, we can avoid those invalid actions by setting the key_value of the action to -99
            # so that the arithmetic won't chose them.
            if location[0] == 0:
                key_value[3] = -99
            if location[0] == 9:
                key_value[2] = -99
            if location[1] == 9:
                key_value[1] = -99
            if location[1] == 0:
                key_value[0] = -99

            key_value = sorted(key_value.items(), key=lambda kv: (kv[1], kv[0]))
            action_list.append(key_value[3][0])
            if key_value[3][1] == key_value[2][1]:
                action_list.append(key_value[2][0])
                if key_value[2][1] == key_value[1][1]:
                    action_list.append(key_value[1][0])
                    if key_value[1][1] == key_value[0][1]:
                        action_list.append(key_value[0][0])

            action = random.choice(action_list)

        else:
            # choose random action to make exploration
            action_list = [0, 1, 2, 3]
            if location[0] == 0:
                del action_list[action_list.index(0)]
            if location[0] == 3:
                del action_list[action_list.index(1)]
            if location[1] == 3:
                del action_list[action_list.index(2)]
            if location[1] == 0:
                del action_list[action_list.index(3)]
            action = random.choice(action_list)
        # action = 2
        return action

    def table_result(self):
        return self.table

