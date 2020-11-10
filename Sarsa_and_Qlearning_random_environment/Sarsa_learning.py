'''
actions:
    0 ----> up
    3 ----> left
    1 ----> down
    2 ----> right
    use this function to transform the x/y location to a single state so that we can reduce the dimensionality
    also
 Map:
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15
'''

from maze_env import Maze
import time
import pandas as pd
import numpy as np
import Sarsa_brain as Model
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
    episode = 1

    greedy_value_renew = 0.9

    global episode_end_flag_list
    if judge_method == 'repeated steps':
        episode_end_flag_list = list(np.zeros(judge_number))

    while True:
        global location_current
        global location_next
        Sarsa_brain_.check_state_exist(str(location_current))
        action_current = Sarsa_brain_.choose_action(location_current)

        # We need to calculate actions for two times
        # First we chose the initial action befor the iteration
        # After the first choice, the program do the step to undate the state of the robot to find the next state
        # Then in the loop, we chose new action based on the new state and the table.

        # initial the location state
        env.reset()
        # Reset the GUI window to the original state (0, 0)
        # Renew the GUI window
        episode_step = 1
        while True:
            env.update()
            # Renew the GUI window
            time.sleep(delay_time)
            # 选取当前状态下的动作
            # print(f'新的循环 location_current:{location_current}')
            location_next, reward_current, complete_flag = env.step(action_current)
            # print(f'执行动作后的位置:{location_next}')

            Sarsa_brain_.check_state_exist(str(location_next))
            # print(f'结束标志{complete_flag}')
            action_next = Sarsa_brain_.choose_action(location_next)
            # When the size of the table has changed, change the threshold in the function chose action.

            Sarsa_brain_.table_calculate_(location_current, action_current, location_next, action_next, reward_current,
                                 step_info=episode_step)
            '''
            Calculate the table value based on the current state, action, reward and
            next state, action
            '''
            action_current = action_next
            location_current = location_next
            episode_step += 1
            reward_info.append(reward_current)
            if complete_flag:
                break
        episode += 1
        # Collect the plot information and result.
        # The index -1 means the last value of the episode. Usually -1 or 1.
        plot_reward.append(reward_info[-1])
        plot_episode.append(episode)
        plot_sum_reward.append(sum(reward_info))
        # If the average of the reward in a sequence of episodes > 0.4, we label it as a kind of converge.
        # The reward of very last state will stop at 0.5
        # Using [episode % judge_number] to serve as the index of the episode_end_list to renew the list over the time.
        # If there are 10 new state and there average over 0.9, we labeled the agent to be converged.
        # We find the value '0.9' based on the total number of episode judgement.
        # Before the program, we can run it in a large number of episodes such as 10000 to find the threshold value.
        # But in this case and q learning, it is just 1. In the MonteCarlo method, this method will work better.

        if reward_info[-1] == 1:
            greedy_value_renew += 0.0001
            Sarsa_brain_.change_greedy(greedy=greedy_value_renew)

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
        # Clear the information in this episode. Extremely important step.
    # env.destroy()


def s_test(judge_number):
    # This function is designed to be imported in other file to carried out test and multiple Sarse learning cases.
    # Such as run the Sarsa learning 20 times to get an average episode number and consuming time.
    global Sarsa_brain_
    global env
    env = Maze(height=10, width=10)
    Sarsa_brain_ = Model.SARSA(greedy_rate=0.9, learning_rate=0.01, reward_decay=0.9)
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

# This 'if' judgement is very special, the function is: if we import this file in other places,
# Rather than running this file directly, the content below will not be processed.


if __name__ == "__main__":
    print('-----------Start-------------')
    env = Maze(height=10, width=10)
    Sarsa_brain_ = Model.SARSA(greedy_rate=0.8, learning_rate=0.01, reward_decay=0.9)
    # Use two methods to evaluate the algorithm,
    # the time it takes to complete the appointed number of episode
    # the step it takes to converge (judged by the repeated steps: if the reward in 10 sequent episodes are the same
    # We roughly think the algorithm started to converge
    # When the judge_method == numbers of repeated steps, the
    # update(judge_number=100, judge_method='numbers of repeated steps')

    T1 = time.perf_counter()
    # update(judge_number=1000, judge_method='sum of episodes', delay_time=0.00)
    update(judge_number=10, judge_method='repeated steps', delay_time=0.00)
    T2 = time.perf_counter()

    print('Time spend :%s ms' % ((T2 - T1)*1000))
    result_display.result_plot(x=plot_episode, y=plot_sum_reward, x_name='Episode', y_name='Sum of Reward',
                               title='Sarsa Learning')
    np.set_printoptions(threshold=len(Sarsa_brain_.table_result()))
    # Some times the data can not be shown cause there are two many of them. Default of the threshold is 1000
    print(Sarsa_brain_.table_result())
    print(f'episode:{max(plot_episode)}')
    print('-----------End of All-------------')
    env.mainloop()




