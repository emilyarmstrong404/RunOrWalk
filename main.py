import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


df = pd.read_csv('dataset.csv')

df.drop(columns=['date', 'time', 'username', 'wrist'], inplace=True)

X = df[["acceleration_x","acceleration_y","acceleration_z","gyro_x","gyro_y","gyro_z"]]
Y = df[["activity"]].values.flatten()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=10)

neural_model = tf.keras.Sequential()
neural_model.add(tf.keras.Input(shape=(6,)))
neural_model.add(tf.keras.layers.Dense(10, activation='relu'))
neural_model.add(tf.keras.layers.Dense(5, activation='relu'))
neural_model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #sigmoid as binary output (walk or run)

neural_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy as binary classification

train_history = neural_model.fit(X_train, Y_train, epochs=100, batch_size=32)

loss, accuracy = neural_model.evaluate(X_test, Y_test)
print(f'Test model loss: {round(loss,4)}')
print(f'Test model accuracy: {round(100*accuracy,2)}%')