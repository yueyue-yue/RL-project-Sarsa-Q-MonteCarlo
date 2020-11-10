
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








