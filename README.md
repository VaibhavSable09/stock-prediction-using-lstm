implementin code in 12 steps these are :
1)Import Libraries
2)Load and Preprocess Data
3)Create Dataset Function
4)Set the Time Step and Create the Dataset
5)Train-Test Split
6)Reshape Data for LSTM
7)Build the LSTM Model with Dropout and Learning Rate Scheduling
8)Train the Model with Learning Rate Scheduling
9)Test Data Preparation
10)Ensure Consistent Lengths for Visualization
11)Create DataFrame for 'valid' and Add 'Predictions' Column
12)Visualize Predictions

#1)Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler


# Load data
data = pd.read_csv('AAPL.csv')
data = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)


# Create dataset function
def create_dataset(dataset, time_step=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        data_x.append(a)
        data_y.append(dataset[i + time_step, 0])
    return np.array(data_x), np.array(data_y)


# Set the time step
time_step = 100

# Create dataset
X, y = create_dataset(data_normalized, time_step)


# Split the data into training and testing sets
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train_data, test_data = data_normalized[0:train_size, :], data_normalized[train_size:len(data), :]


# Reshape the input for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)


# Define a learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch > 0:
        lr = lr * 0.9
    return lr

# Build the LSTM model with dropout
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


# Train the model with learning rate scheduling
model.fit(X, y, epochs=20, batch_size=32, callbacks=[LearningRateScheduler(lr_scheduler)])


# Test data preparation
test_data_len = len(test_data)
x_test, y_test = create_dataset(test_data, time_step)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# Evaluate model performance (Root Mean Squared Error)
rmse = np.sqrt(np.mean(np.power((test_data[time_step:] - predictions), 2)))
print(f'Root Mean Squared Error: {rmse}')


# Step 9: Test Data Preparation
test_data_len = len(test_data)
x_test, y_test = create_dataset(test_data, time_step)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Ensure the lengths of test_data and predictions are consistent
test_data_for_plot = np.empty(len(data))
test_data_for_plot[:] = np.nan
test_data_for_plot[len(data) - len(test_data):] = test_data.flatten()

predictions_for_plot = np.empty(len(data))
predictions_for_plot[:] = np.nan
predictions_for_plot[len(data) - len(predictions):] = predictions.flatten()

# Create DataFrame for 'valid' and add 'Predictions' column
valid_df = pd.DataFrame({
    'Actual': test_data_for_plot,
    'Predictions': predictions_for_plot
})




# Create DataFrame for 'valid' and add 'Predictions' column
valid_df = pd.DataFrame({
    'Actual': test_data_for_plot,
    'Predictions': predictions_for_plot
})


# ... Existing code ...

# Step 12: Visualize Predictions
plt.figure(figsize=(16, 8))
plt.title('Model Evaluation')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')

# Plotting Actual Closing Prices
plt.plot(valid_df['Actual'], label='Actual', color='blue')

# Plotting Predictions
plt.plot(valid_df['Predictions'], label='Predictions', color='orange')

plt.legend()
plt.show()






# Additional Cell 2: Scatter Plot of Actual vs. Predicted Values
plt.figure(figsize=(10, 8))
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.scatter(valid_df['Actual'], valid_df['Predictions'], color='purple', alpha=0.7)
plt.xlabel('Actual Close Price USD ($)')
plt.ylabel('Predicted Close Price USD ($)')
plt.show()

# Additional Cell 1: Histogram of Actual and Predicted Values
plt.figure(figsize=(12, 6))
plt.title('Histogram of Actual and Predicted Values')
plt.hist(valid_df['Actual'], bins=30, alpha=0.5, label='Actual', color='blue')
plt.hist(valid_df['Predictions'], bins=30, alpha=0.5, label='Predictions', color='orange')
plt.xlabel('Close Price USD ($)')
plt.ylabel('Frequency')
plt.legend()
plt.show()



