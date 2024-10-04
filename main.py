import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')

df[["acceleration_x","acceleration_y","acceleration_z","gyro_x","gyro_y","gyro_z"]] = df[["acceleration_x","acceleration_y","acceleration_z","gyro_x","gyro_y","gyro_z"]].astype(np.float32)
df["activity"] = df["activity"].astype(np.float32)

X = df[["acceleration_x","acceleration_y","acceleration_z","gyro_x","gyro_y","gyro_z"]]
Y = df[["activity"]]

b_init = 0
w_init = np.zeros(6)
iterations = 1000
learning_rate = 0.001

def compute_cost(x, y, w, b):
    m = x.shape[0]
    predictions = np.dot(x, w) + b
    errors = predictions - y.values.flatten()
    total_cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return total_cost

def compute_gradient(x, y, w, b):
    x = x.values
    y = y.values
    m = x.shape[0]
    predictions = np.dot(x, w) + b
    errors = predictions - y.flatten()
    dj_dw = (1 / m) * np.dot(x.T, errors)
    dj_db = (1 / m) * np.sum(errors)
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, gradient_function, alpha, num_iters):
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}, Cost: {cost}")

    return w, b

w,b = gradient_descent(X ,Y, w_init, b_init, compute_gradient, learning_rate, iterations)
print("w,b found by gradient descent, w: ", w, "b ", b)

m = X.shape[0]
accuracy_counter = 0
for i in range(m):
    if np.dot(X.iloc[i], w + b) < 0.5:
        walk_or_run = "walk"
    else:
        walk_or_run = "run"

    if int(Y.iloc[i]) < 0.5:
        target_walk_or_run = "walk"
    else:
        target_walk_or_run = "run"
    print(f"prediction: {np.dot(X.iloc[i], w + b):0.2f}, target value: {Y.iloc[i]}, prediction: {walk_or_run}, target: {target_walk_or_run}" )
    if walk_or_run == target_walk_or_run:
        accuracy_counter += 1
print("Accuracy: ", accuracy_counter/m)

