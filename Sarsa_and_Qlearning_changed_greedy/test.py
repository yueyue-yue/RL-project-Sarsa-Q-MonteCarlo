import Qlearning as QL
import Sarsa_learning as SAR
import result_display as dp
import matplotlib.pyplot as plt

# print(QL.q_test(20))
# print(SAR.s_test(20))
running_time_M_ = []
running_time_S_ = []
running_time_Q_ = []
episode_M_ = []
episode_S_ = []
episode_Q_ = []
plot_sum_reward_M_ = []
plot_sum_reward_S_ = []
plot_sum_reward_Q_ = []
episode_M, episode_S, episode_Q = 0, 0, 0

train_volumn = 30
for i in range(train_volumn):

    '''
    I don't know why there is always bugs when running MonteCarlo in this file and try to fix the bug but failed.
    The test about MonteCarlo is done in the MonteCarlo_learning file. I hope you can understand that. 
    I have tried even several days to rewrite the code but still didn't work. 
    
    running_time_M, episode_M, plot_sum_reward_M = MON.m_test(20)
    episode_M_.append(episode_M)
    running_time_M_.append(running_time_M)
    plot_sum_reward_M_.append(plot_sum_reward_M)
    '''

    running_time_S, episode_S, plot_sum_reward_S = SAR.s_test(20)
    episode_S_.append(episode_S)
    running_time_S_.append(running_time_S)
    plot_sum_reward_S_.append(plot_sum_reward_S)
    
    running_time_Q, episode_Q, plot_sum_reward_Q = QL.q_test(20)
    episode_Q_.append(episode_Q)
    running_time_Q_.append(running_time_Q)
    plot_sum_reward_Q_.append(plot_sum_reward_Q)

Sarsa_run_time = sum(running_time_S_)/train_volumn
average_S_episode = sum(episode_S_)/train_volumn

Q_run_time = sum(running_time_Q_)/train_volumn
average_Q_episode = sum(episode_Q_)/train_volumn

print(f'Sarsa_run_time:{Sarsa_run_time},Q_run_time:{Q_run_time}')
print(f'running_time_S_:{running_time_S_}')
print(f'running_time_Q_:{running_time_Q_}')
# print(f'sum_reward_M:{plot_sum_reward_M_},sum_reward_S:{plot_sum_reward_S_},sum_reward_Q:{plot_sum_reward_Q_}')
print(f'episode_S_:{episode_S_}')
print(f'episode_Q_:{episode_Q_}')
print(f'average_S_episode:{average_S_episode},average_Q_episode:{average_Q_episode}')



plt.plot(range(len(episode_S_)), episode_S_, 'bD-', episode_Q_, 'r^-')
plt.legend(('Sarsa learning', 'Q learning'), loc='upper right')
plt.xlabel('Times')
plt.ylabel('Converged Episodes')
plt.title('Times-Converged Episodes')
plt.show()

# return running_time, episode, plot_sum_reward

# dp.result_plot(range(1, episode_S_[0]), plot_sum_reward_S_[0], 'Episode', 'reward', 'Mon')

# x, y, x_name, y_name, title




