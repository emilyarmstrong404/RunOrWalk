import pandas as pd
from sklearn import linear_model

df = pd.read_csv('dataset.csv')

X = df[["acceleration_x","acceleration_y","acceleration_z","gyro_x","gyro_y","gyro_z"]]
Y = df[["activity"]]

regressor = linear_model.LinearRegression()
regressor.fit(X, Y)

test_data = pd.DataFrame([[0.5972, -1.1432, -0.2235, 0.4744, -1.1092, -2.9296]],
                         columns=["acceleration_x", "acceleration_y", "acceleration_z", "gyro_x", "gyro_y", "gyro_z"])
prediction = regressor.predict(test_data)

if prediction < 0.5:
    print("Walk")
elif prediction >= 0.5 and prediction <= 1:
    print("Run")
else:
    print("Error")

print(prediction)

