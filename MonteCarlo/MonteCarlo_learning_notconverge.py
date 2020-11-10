'''
actions:
    0 ----> up
    3 ----> left
    1 ----> down
    2 ----> right
'''

from maze_env import Maze
import time
import pandas as pd
import numpy as np
import MonteCarlo_brain as Model
import matplotlib as mpl
import matplotlib.pyplot as plt
import result_display

location_current = [0, 0]
location_next = [0, 0]
action_current = 0

location_info = []
action_info = []
reward_info = []

plot_reward = []
plot_episode = []
plot_sum_reward = []
episode_end_flag_list = []


def first_visit_data_process(location_info, action_info, reward_info):
    # Before we process the calculation, we del those data appear more than once.
    if len(location_info) <= 1:
        return location_info, action_info, reward_info
    for i in range(len(location_info) - 1, 0, -1):
        # del the same element according to the state
        for j in range(0, i):
            # print(f'a[i]:{location_info[i]},a[j]:{location_info[j]}')
            if location_info[i] == location_info[j]:
                del location_info[i]
                del action_info[i]
                del reward_info[i]
                break
    return location_info, action_info, reward_info


def update(judge_number, judge_method, delay_time):
    episode = 1

    global episode_end_flag_list
    if judge_method == 'repeated steps':
        episode_end_flag_list = list(np.zeros(judge_number))

    while True:
        # print('-------------------Episode Begin-------------------')
        global location_current
        global location_next
        global location_info
        global action_info
        global reward_info
        MonteCarlo_brain_.check_state_exist(str(location_current))
        action_current = MonteCarlo_brain_.choose_action(location_current)
        # We need to calculate actions for two times
        # First we chose the initial action before the iteration
        # After the first choice, the program do the step to update the state of the robot to find the next state
        # Then in the loop, we chose new action based on the new state and the table.
        # initial the location state
        env.reset()
        # Reset the GUI window to the original state (0, 0)
        # Renew the GUI window
        episode_step = 1
        stop_delay_1_episode = 0
        while True:
            env.update()
            # Renew the GUI window
            time.sleep(delay_time)
            # 选取当前状态下的动作
            # print(f'新的循环 location_current:{location_current}')
            location_next, reward_current, complete_flag = env.step(action_current)
            # print(f'执行动作后的位置:{location_next}')
            # print(f'实时奖励：{reward_current}')
            MonteCarlo_brain_.check_state_exist(str(location_next))
            # print(f'结束标志{complete_flag}')
            action_next = MonteCarlo_brain_.choose_action(location_next)
            action_next = 2
            # When the size of the table has changed, change the threshold in the function chose action.

            # 更新状态 进入下一次的循环
            # At the end of the loop, we use new state, action to replace old ones to complete the iteration
            # print(f'location_current:{location_current},location next:{location_next}')
            # print(f'现在位置:{location_current},下一位置:{location_next}')
            # print(f'现在动作:{action_current},下一动作:{action_next}')
            location_info.append(location_current)
            action_info.append(action_current)
            action_current = action_next
            location_current = location_next
            episode_step += 1
            # print(f'奖励信息{reward_info}')
            '''
            if complete_flag:
                reward_info.append(reward_current)
                stop_delay_1_episode = 1
                continue
            if stop_delay_1_episode == 1:
                break
            '''
            print('reward_current', reward_current)
            if complete_flag:
                reward_info.append(reward_current)
                break
            reward_info.append(-0.04)
        episode += 1

        # Before we calculate, delete those repeated (s,a) state according to first visit principle.
        # Use reversed list to renew the MonteCarlo table and renew the table.
        # The renew function is attached inside the  'MonteCarlo_brain_.table_calculate_()'
        # Calculate the table value based on the current state, action, reward and next state, action
        print(f'pre-process '
              f'location_info:{location_info},action_info:{action_info},reward_info:{reward_info}')
        location_info, action_info, reward_info = first_visit_data_process(location_info, action_info, reward_info)
        print(f'location_info:{location_info},action_info:{action_info},reward_info:{reward_info}')

        # This is a special bug. At last of the action, our next state is goal or hole.
        # But they are not append in our location_info and action_info list.
        # We take use lase_step parameter to let the g_value equals 0 according to MonteCarlo algorithm.
        for i in range(len(reward_info) - 1, -1, -1):
            # print('i:', i)
            if i == (len(reward_info) - 1):
                location_current = location_info[i]
                action_current = action_info[i]
                reward_current = reward_info[i]
                location_next = [0, 0]
                action_next = 1
                MonteCarlo_brain_.table_calculate_(location_current, action_current, location_next,
                                                   action_next, reward_current, last_step=True)
            else:
                location_current = location_info[i]
                action_current = action_info[i]
                reward_current = reward_info[i]
                location_next = location_info[i + 1]
                action_next = action_info[i + 1]
                MonteCarlo_brain_.table_calculate_(location_current, action_current, location_next,
                                                   action_next, reward_current, last_step=False)

        # Collect the plot information and result.
        plot_reward.append(reward_info[-1])
        plot_episode.append(episode)
        plot_sum_reward.append(sum(reward_info))

        # If the average of the reward in a sequence of episodes > 0.4, we label it as a kind of converge.
        # The reward of very last state will stop at 0.5
        if judge_method == 'repeated steps':
            episode_end_flag_list[episode % judge_number] = sum(reward_info)
            episode_end_flag = sum(episode_end_flag_list)
            if episode_end_flag >= 0.4*judge_number:
                break
        elif judge_method == 'sum of episodes':
            if episode >= judge_number:
                break
        else:
            print('Error episode end judge methods input!!!!!!!!!!!!!!')
            break
        location_info.clear()
        action_info.clear()
        reward_info.clear()
    # print('--------------------Episode End----------------------')
    # env.destroy()
    # This is to destroy the window of the explorer.


def m_test(judge_number):
    global MonteCarlo_brain_
    global env
    env = Maze(height=10, width=10)
    MonteCarlo_brain_ = Model.Monte(greedy_rate=0.5, learning_rate=0.9, reward_decay=0.9)
    T1 = time.perf_counter()
    update(judge_number=judge_number, judge_method='repeated steps', delay_time=0.00)
    T2 = time.perf_counter()
    running_time = (T2 - T1) * 1000
    episode = max(plot_episode)
    env.destroy()
    env.mainloop()
    return running_time, episode, plot_sum_reward


# This 'if' judgement is very special, the function is that if we import this file in other places,
# Rather than running this file directly, the content below will not be processed.

if __name__ == "__main__":
    print('-----------Start-------------')
    env = Maze(height=10, width=10)
    MonteCarlo_brain_ = Model.Monte(greedy_rate=0.9, learning_rate=0.9, reward_decay=0.9)
    # Use two methods to evaluate the algorithm,
    # the time it takes to complete the appointed number of episode
    # the step it takes to converge
    # (The converge is judged by the repeated steps: if reward in 10 sequent episodes beyond certain value)
    # The certain value is get by run 10000 episodes of it. (judge_method = 'sum of episodes')

    # When the judge_method == repeated steps, the agent will stop when the episode has judge_number of sequent episode
    # satisfied the converge condition.
    # When the judge_method == sum of episodes, the agent will stop when the episode reach the judge number.

    # The delay_time parameter in the update is set aims to show us the action of the agent if we want.(0.01 is enough)

    T1 = time.perf_counter()
    update(judge_number=1, judge_method='sum of episodes', delay_time=0.00)
    # update(judge_number=6, judge_method='repeated steps', delay_time=0.00)
    T2 = time.perf_counter()
    print('Time spend :%s ms' % ((T2 - T1)*1000))
    result_display.result_plot(x=plot_episode, y=plot_sum_reward, x_name='Episode', y_name='Sum of Reward',
                               title='MonteCarlo Learning')
    print(MonteCarlo_brain_.table_result())
    print(f'episode:{max(plot_episode)}')
    print('-----------End of All-------------')
    env.mainloop()


