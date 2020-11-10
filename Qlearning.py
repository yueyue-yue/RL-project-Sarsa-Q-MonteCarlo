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
import Q_brain as Model
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


def update(judge_number, judge_method, delay_time):
    # pay attention if using
    episode = 1

    greedy_value_renew = 0.9

    global episode_end_flag_list
    if judge_method == 'repeated steps':
        episode_end_flag_list = list(np.zeros(judge_number))

    while True:
        global location_current
        global location_next
        env.reset()
        Q_brain_.check_state_exist(str(location_current))
        action_current = Q_brain_.choose_action(location_current)
        # We need to calculate actions for two times
        # First we chose the initial action befor the iteration
        # After the first choice, the program do the step to undate the state of the robot to find the next state
        # Then in the loop, we chose new action based on the new state and the table.
        # initial the location state
        # Reset the GUI window to the original state (0, 0)
        # Renew the GUI window
        episode_step = 1
        # time.sleep(100)
        while True:
            env.update()
            # Renew the GUI window
            time.sleep(delay_time)
            # 选取当前状态下的动作
            # print(f'新的循环 location_current:{location_current}')
            location_next, reward_current, complete_flag = env.step(action_current)
            # print(f'执行动作后的位置:{location_next}')

            Q_brain_.check_state_exist(str(location_next))
            # print(f'结束标志{complete_flag}')
            action_next = Q_brain_.choose_action(location_next)
            # When the size of the table has changed, change the threshold in the function chose action.

            Q_brain_.table_calculate_(location_current, action_current, location_next, reward_current,
                                 step_info=episode_step, last_step=complete_flag)
            '''
            Calculate the table value based on the current state, action, reward and
            next state, action
            '''
            # 更新状态 进入下一次的循环
            # At the end of the loop, we use new state, action to replace old ones to complete the iteration

            # print(f'location_current:{location_current},location next:{location_next}')
            # print(f'现在位置:{location_current},complete_flag:{complete_flag}')
            # print(f'现在动作:{action_current},下一动作:{action_next}')
            # print(f'位置：{location_current}'
            #      f'动作：{action_current}')
            action_current = action_next
            location_current = location_next
            episode_step += 1
            reward_info.append(reward_current)
            if complete_flag:
                break

        location_current = [0, 0]
        location_next = [0, 0]
        episode += 1
        # Collect the plot information and result.
        plot_reward.append(reward_info[-1])
        plot_episode.append(episode)
        plot_sum_reward.append(sum(reward_info))
        # print('episode:', episode)
        # If the average of the reward in a sequence of episodes > 0.4, we label it as a kind of converge.
        # The reward of very last state will stop at 0.5
        # Using [episode % judge_number] to serve as the index of the episode_end_list to renew the list over the time.
        # If there are 10 new state and there average over 0.9, we labeled the agent to be converged.
        # We find the value '0.9' based on the total number of episode judgement.
        # Before the program, we can run it in a large number of episodes such as 10000 to find the threshold value.
        # But in this case and q learning, it is just 1. In the MonteCarlo method, this method will work better.

        # Algorithm will become more and more greedy as the robot getting the goal.
        if reward_info[-1] == 1:
            greedy_value_renew += 0.0001
            Q_brain_.change_greedy(greedy=greedy_value_renew)

        if judge_method == 'repeated steps':
            episode_end_flag_list[episode % judge_number] = sum(reward_info)
            episode_end_flag = sum(episode_end_flag_list)
            if episode_end_flag >= 0.9 * judge_number:
                break
        elif judge_method == 'sum of episodes':
            if episode >= judge_number:
                break
        else:
            print('Error judge methods')
            break

        location_info.clear()
        action_info.clear()
        reward_info.clear()
    # env.destroy()

# This function is designed to carry out multiple test and get the average value of the learning.
# By the running_time, episode, we can make comparasion between different algorithm.

def q_test(judge_number):
    global Q_brain_
    global env

    env = Maze(height=10, width=10)
    Q_brain_ = Model.Qlearning(greedy_rate=0.9, learning_rate=0.01, reward_decay=0.9)
    T1 = time.perf_counter()
    update(judge_number=judge_number, judge_method='repeated steps', delay_time=0.00)
    T2 = time.perf_counter()
    running_time = (T2 - T1) * 1000
    episode = max(plot_episode)
    env.destroy()
    env.mainloop()
    plot_episode.clear()
    # This step is a very special bug. We import this function and run this in a for loop.
    # The data in plot_episode will be stored in thr for loop.
    # We must clear it at the end.
    return running_time, episode, plot_sum_reward

# This 'if' judgement is very special, the function is that if we import this file in other places,
# Rather than running this file directly, the content below will not be processed.

#q_test(20)

if __name__ == "__main__":
    print('-----------Start-------------')
    env = Maze(height=10, width=10)
    Q_brain_ = Model.Qlearning(greedy_rate=0.9, learning_rate=0.01, reward_decay=0.9)
    # Use two methods to evaluate the algorithm,
    # the time it takes to complete the appointed number of episode
    # the step it takes to converge (judged by the repeated steps: if the reward in 10 sequent episodes are the same
    # We roughly think the algorithm started to converge
    # When the judge_method == numbers of repeated steps, the
    T1 = time.perf_counter()
    update(judge_number=10, judge_method='repeated steps', delay_time=0.00)
    # update(judge_number=50, judge_method='sum of episodes', delay_time=0.00)
    T2 = time.perf_counter()
    print('Time spend :%s ms' % ((T2 - T1)*1000))
    result_display.result_plot(x=plot_episode, y=plot_sum_reward, x_name='Episode', y_name='Reward',
                               title='Q Learning')
    np.set_printoptions(threshold=len(Q_brain_.table_result()))
    # Some times the data can not be shown cause there are two many of them. Default of the threshold is 1000
    print('0: up 1: down 2: right  3: left')
    print(Q_brain_.table_result())
    print(f'episode:{max(plot_episode)}')
    print('-----------End of All-------------')
    env.mainloop()


