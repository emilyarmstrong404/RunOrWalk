import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# load dataset into dataframe
df = pd.read_csv('dataset.csv')

# remove unused columns of data
df.drop(columns=['date', 'time', 'username', 'wrist'], inplace=True)

# load input and label data into x and y
X = df[["acceleration_x","acceleration_y","acceleration_z","gyro_x","gyro_y","gyro_z"]]
Y = df[["activity"]].values.flatten()

# scaling data using scikit-learn standard scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Split data into 75% training data and 25% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=10)

# building layers of neural network
neural_model = tf.keras.Sequential()
neural_model.add(tf.keras.Input(shape=(6,))) # 6 input variables
neural_model.add(tf.keras.layers.Dense(10, activation='relu'))
neural_model.add(tf.keras.layers.Dense(5, activation='relu'))
neural_model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #sigmoid as binary output (walk or run)

# compiling the neural network using adam optimiser and binary crossentropy loss function
neural_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy as binary classification

# training model over 100 cycles over the dataset, updating the weights every 32 data samples
train_history = neural_model.fit(X_train, Y_train, epochs=100, batch_size=32)

# evaluation of model performance
loss, accuracy = neural_model.evaluate(X_test, Y_test)
print(f'Test model loss: {round(loss,4)}')
print(f'Test model accuracy: {round(100*accuracy,2)}%')