
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def result_plot(x, y, x_name, y_name, title):
    x = x
    y = y
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.show()


Sarsa_run_time = 5705.537523333333
Q_run_time = 1219.9195433333307
running_time_S_ = [17503.029899999998, 2015.7580999999993, 2090.9975000000004, 3406.336, 3104.112900000004,
                   666.0809000000042, 6666.408799999999, 7448.578300000001, 4264.321299999999, 1257.1589000000004,
                   8561.543799999996, 24460.007199999993, 1159.5603999999894, 9075.184799999988, 1633.5658999999937,
                   1213.1953000000094, 2127.5978000000036, 9033.096100000008, 8948.680800000006, 1607.0100999999966,
                   1843.5390999999868, 9473.086499999994, 3083.6014000000205, 12200.408300000021, 2742.2933000000003,
                   1085.2711, 5132.539800000018, 7998.425799999978, 2965.884799999998, 8398.85080000002]
running_time_Q_ = [1689.7767999999971, 1099.4048999999998, 1328.656500000001, 1370.1339000000025, 1404.9627999999998,
                   1012.4844999999993, 539.8944999999956, 1106.7524000000049, 2323.4430999999986, 812.986599999995,
                   1475.549600000008, 862.3037999999923, 1523.5370999999986, 1940.8418999999953, 1750.5950000000041,
                   1200.970599999991, 1712.0575999999942, 504.893299999992, 1001.6845000000103, 1726.0195999999723,
                   1118.3235000000025, 750.9562000000187, 1343.3597000000077, 1474.2411000000004, 384.98439999997913,
                   1014.2448000000002, 1682.3189999999784, 765.1784999999904, 1110.9247999999923, 566.1053000000038]
episode_S_ = [1245, 230, 262, 285, 335,
              98, 678, 556, 452, 146,
              887, 1828, 101, 792, 170,
              150, 223, 843, 767, 180,
              171, 912, 320, 625, 253,
              122, 370, 731, 275, 693]
episode_Q_ = [306, 194, 186, 241, 206,
              178, 64, 150, 376, 83,
              215, 101, 231, 305, 215,
              194, 246, 73, 116, 298,
              165, 99, 181, 224, 48,
              139, 293, 83, 152, 83]
average_S_episode = 490.0
average_Q_episode = 181.5


running_time_M_ = [10356.2236, 2115.5341000000003, 1500.4851, 8885.498500000002, 1532.0192000000002,
                   5953.0274, 910.9947000000001, 2690.1762000000003, 5494.3444, 1093.2984999999999,
                   7835.6840999999995, 679.7105, 1553.3236000000002, 22415.1919, 2782.7643000000003,
                   2783.7134, 24430.848, 1691.456, 1947.6829000000002, 666.7122,
                   1369.3822999999998, 4448.8247, 6573.2729, 2229.6072, 1402.5460999999998,
                   1205.4102, 1284.0892000000001, 2261.5265999999997, 2752.3276, 1129.0013000000001]

episode_M_ = [1347, 256, 179, 773, 546,
              912, 178, 276, 931, 165,
              1265, 3688, 329, 3045, 260,
              291, 2547, 206, 250, 73,
              180, 405, 875, 282, 193,
              173, 175, 221, 268, 151]


# ---------------------------------------------------------------------------------------------------------------------#
# Below are the data come from changed greedy poilcy.
'''
running_time_S_:[1635.0034999999998, 5689.278399999999, 4000.5284, 4079.5765999999994, 2092.4022999999997, 
                 30140.5775, 1154.426200000003, 16679.672499999997, 5775.159000000002, 11194.777300000012, 
                 1053.8523000000027, 4091.0530999999964, 15343.800399999993, 4446.286500000014, 3963.2231999999876, 
                 6465.030899999988, 17439.559, 7301.775400000025, 2529.18870000002, 1306.7293999999947, 
                 1969.4434999999828, 4500.067400000006, 6122.899999999987, 2328.6760000000013, 30365.94720000002, 
                 8654.000700000011, 17667.669100000014, 4843.149700000026, 5216.271100000029, 7969.984000000011]
                 
running_time_Q_:[819.5389, 2360.0390000000007, 1556.559, 1196.3223999999996, 1297.1503999999995, 
                 1299.351299999998, 1236.7254000000046, 1830.7179999999903, 2043.172400000003, 2175.8066999999955, 
                 1489.9300000000012, 2527.002600000003, 1136.5724999999998, 1316.7913000000055, 1703.1502000000103, 
                 1319.1162000000247, 1001.3025000000084, 2790.9985999999944, 921.1795999999879, 2209.976499999982, 
                 1790.5022000000201, 1098.3879000000059, 1139.0591999999913, 1352.2347999999909, 2284.563599999984, 
                 2021.825999999976, 2141.1600000000135, 1309.847599999955, 1928.683699999965, 1342.685700000004]
episode_S_:[104, 373, 321, 305, 155, 
            1402, 77, 879, 340, 540, 
            104, 282, 565, 329, 292, 
            379, 804, 434, 247, 112, 
            136, 192, 360, 222, 1088, 
            380, 1000, 347, 414, 365]
            
episode_Q_:[86, 308, 171, 93, 120, 
            154, 99, 202, 188, 176, 
            138, 177, 122, 153, 205, 
            156, 94, 210, 96, 185, 
            216, 87, 110, 120, 219, 
            193, 257, 141, 235, 111]
average_S_episode:418.26666666666665,average_Q_episode:160.73333333333332
'''
run_time_S_improved = [1635.0034999999998, 5689.278399999999, 4000.5284, 4079.5765999999994, 2092.4022999999997,
                  30140.5775, 1154.426200000003, 16679.672499999997, 5775.159000000002, 11194.777300000012,
                  1053.8523000000027, 4091.0530999999964, 15343.800399999993, 4446.286500000014, 3963.2231999999876,
                  6465.030899999988, 17439.559, 7301.775400000025, 2529.18870000002, 1306.7293999999947,
                  1969.4434999999828, 4500.067400000006, 6122.899999999987, 2328.6760000000013, 30365.94720000002,
                  8654.000700000011, 17667.669100000014, 4843.149700000026, 5216.271100000029, 7969.984000000011]

run_time_Q_improved = [819.5389, 2360.0390000000007, 1556.559, 1196.3223999999996, 1297.1503999999995,
                  1299.351299999998, 1236.7254000000046, 1830.7179999999903, 2043.172400000003, 2175.8066999999955,
                  1489.9300000000012, 2527.002600000003, 1136.5724999999998, 1316.7913000000055, 1703.1502000000103,
                  1319.1162000000247, 1001.3025000000084, 2790.9985999999944, 921.1795999999879, 2209.976499999982,
                  1790.5022000000201, 1098.3879000000059, 1139.0591999999913, 1352.2347999999909, 2284.563599999984,
                  2021.825999999976, 2141.1600000000135, 1309.847599999955, 1928.683699999965, 1342.685700000004]
episode_S_improved = [104, 373, 321, 305, 155,
                      1402, 77, 879, 340, 540,
                      104, 282, 565, 329, 292,
                      379, 804, 434, 247, 112,
                      136, 192, 360, 222, 1088,
                      380, 1000, 347, 414, 365]

episode_Q_improved = [86, 308, 171, 93, 120,
             154, 99, 202, 188, 176,
             138, 177, 122, 153, 205,
             156, 94, 210, 96, 185,
             216, 87, 110, 120, 219,
             193, 257, 141, 235, 111]
plt.plot(range(len(episode_S_)), episode_S_, 'bD-', episode_S_improved, 'r^-')

# plt.plot(range(len(running_time_Q_)), running_time_Q_, 'bD-', run_time_Q_improved, 'r^-')

plt.legend(('Sarsa learning', 'Sarsa learning with dynamic greedy'), loc='upper right')
plt.xlabel('Times')
plt.ylabel('Converged Episodes')
plt.title('Times-Converged Episodes')
plt.show()
print('average episodes M:', sum(episode_M_)/30)
print('average M time:', sum(running_time_M_)/30)








