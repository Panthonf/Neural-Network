from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define the neural network architecture
input_nodes = 9
hidden_nodes = 9
output_nodes = 2

# Initialize the model
model = Sequential()

# Add layers to the model
model.add(Dense(hidden_nodes, input_dim=input_nodes, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_nodes, activation='relu'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(
    learning_rate=0.01), metrics=['accuracy'])

data2 = pd.read_csv(
    r'C:\Users\Lenovo\Desktop\Term 2-2565\Data Mining - 261448\Hw3 NN\train1.txt', header=None)

arrInput = []
arrOutput = []

# Define the training data
for i in range(data2.__len__()):
    arrInput.append([data2[1][i], data2[2][i], data2[3][i],
                     data2[4][i], data2[5][i], data2[6][i], data2[7][i], data2[8][i], data2[9][i]])
    arrOutput.append([data2[0][i]])

# Define the training data
input_data = np.array(arrInput)
output_data = np.array(arrOutput)
output_data = output_data.reshape((-1, 1))


# Train the model
model.fit(input_data, output_data, epochs=100, batch_size=32)
model.summary()

# Test the model
predictions = model.predict(input_data)

for i, prediction in enumerate(predictions):
    print("Input: {}, Target: {}, Output: {}".format(
        input_data[i], output_data[i], float(prediction)))

accuracy = model.evaluate(input_data, output_data, verbose=0)
accuracy[0] = accuracy[0]*100
accuracy[1] = accuracy[1]*100
print(f'Train Accuracy: {accuracy[1]} %')
print(f'Train Loss: {accuracy[0]} %')
