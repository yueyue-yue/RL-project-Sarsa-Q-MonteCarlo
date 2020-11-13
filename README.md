# RL-project-Sarsa-Q-MonteCarlo
Class project recording
-----------------------------------------------------------------
The program have two parts: 4*4 environment and 10*10 environment.
There are still some bugs in monte_carlo_state_transfer. Remaining the file
want to illustrate it is a bad idea to use two coordinates in the program.
-----------------------------------------------------------------
Numpy, time, sys, random, tkinter, matplotlib packages are needed.
Please do not Change the Place File located with.
The packages inside a folder will be needed and are different in different folders even their name are the same.
-----------------------------------------------------------------
maze_env is the environment creation file, consisting of a class
with init, build, reset and step functions.

X_brain files mainly consist of a class with init, check state,
change_greedy, table calculate and table renew functions.

X_learning files consist of update function.
(Special functions are added due to differences between algorithms.)

Test files are used to doing multiple learning process.
result_display files are used to plot and show the results.
-----------------------------------------------------------------
When doing verification, run the x_learning files.
(Q learning is always the best one.)
If want to see multiple results, run the test files.
Initial holes generating policy is DFS policy in 'Sarsa_and_Qlearning_random_environment'.
(Can be changed in maze_env.py in that folder.)
-----------------------------------------------------------------
-----------------------------------------------------------------
Yue Zenglin 岳增霖 2020.10 Mail Address e0575902@u.nus.edu
