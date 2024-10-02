import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')

X = df[["acceleration_x","acceleration_y","acceleration_z","gyro_x","gyro_y","gyro_z"]]
Y = df[["activity"]]

b_init = 0
w_init = np.zeros(6)
iterations = 1000
learning_rate = 0.001

def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    i = 0
    for i in range(m):
        unsqcost = w * x[i] + b - y[i]
        sqcost = unsqcost * unsqcost
        total_cost += sqcost
    total_cost = total_cost / (2 * m)
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    i = 0
    for i in range(m):
        prediction = w * x[i] + b
        dj_db += prediction - y[i]
        dj_dw += (prediction - y[i]) * x[i]
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db