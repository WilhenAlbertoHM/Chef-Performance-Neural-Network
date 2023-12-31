import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Read the dataset file chef_data.pickle
with open("data/chef_data.pickle", "rb") as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f)

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# Define the Keras Sequential model
model = Sequential()

# Use MinMaxScaler to scale y-values before using them for 
# training/testing with same scale factor
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.fit_transform(y_test)

# Build model with an input and an ouput layer (no hidden layers)
# 4 output and 20 input units, with linear activation function
model.add(Dense(units=4, input_dim=20, activation="linear"))

# Compile the Keras model with MSE and Adam optimizer
model.compile(loss="mse", optimizer="adam")

# Print summary of model
print("|==============================| Summary |==============================|")
model.summary()

# Train model with 50 epochs
num_epochs = 50
model.fit(x_train, y_train_scaled, epochs=num_epochs)

# Evaluate the model by predicting the outputs and rescale them back to see actual predictions
y_pred_scaled = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Calculate mean and standard deviation of the difference between predictions and actual values 
# across all test samples for each of the four variables to get an idea of how well it works.
# Take mean and standard deviation over axis = 0.
y_diff = np.abs(np.subtract(y_pred, y_test))
avgs = np.average(y_diff, axis=0)
std_devs = np.std(y_diff, axis=0)

# Evaluate the test and train losses.
train_loss = model.evaluate(x_train, y_train_scaled)
test_loss = model.evaluate(x_test, y_test_scaled)

# Print the number of epochs, activation function, and the loss function used.
print()
print("|================================================================================|")
print("Number of epochs used: ", num_epochs)
print("Activation function used: linear")
print("Loss function used: MSE")

# Print final MSE and accuracy for test and train losses.
print()
print("|================================================================================|")
print("Train MSE loss: ", train_loss)
print("Test MSE loss: ", test_loss)

# Print the averages and standard deviations
print()
print("|================================================================================|")
print("Averages of the variables: ", avgs)
print("Standard deviation values of the variables", std_devs)

# Print the outputs and check the averages with its standard deviation
print()
print("|================================================================================|")
print("Number of tasks completed: ", str(round(avgs[0], 2)) + " +- " + str(round(std_devs[0], 2)) + " -> ~" + str(round(avgs[0])) + " task(s)")
print("Number of tasks not completed: ", str(round(avgs[1], 2)) + " +- " + str(round(std_devs[1], 2)) + " -> ~" + str(round(avgs[1])) + " task(s)")
print("Average bonus ($) each chef earned after the project: ", str(round(avgs[2], 2)) + " +- " + str(round(std_devs[2], 2)))
print("Money ($) that the company earned/lost after the project: ", str(round(avgs[3], 2)) + " +- " + str(round(std_devs[3], 2)))
print()