import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
from IPython.display import display, HTML
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

epochs = 100

# Function to plot training loss over epochs with detailed title
def plot_training_history(history, model_name, data_description):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'Training Loss over Epochs ({model_name}) - {data_description}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
# %% 
# Load the dataset
file_path = 'household_power_consumption.txt'  # Update this path if necessary
data = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False)

# Initial data overview
print("Initial Data Overview:")
display(data.head())

# Summary statistics
print("\nSummary Statistics:")
display(data.describe())

# Get the information about the dataframe
print("\nInformation about the dataframe:")
display(data.info())


# Convert data types and handle missing values
data.replace('?', np.nan, inplace=True)
data = data.astype({'Global_active_power': 'float64',
                    'Global_reactive_power': 'float64',
                    'Voltage': 'float64',
                    'Global_intensity': 'float64',
                    'Sub_metering_1': 'float64',
                    'Sub_metering_2': 'float64',
                    'Sub_metering_3': 'float64'})

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values per Column:")
display(missing_values)
#####################################

# Visualize missing values
file_path = 'household_power_consumption.txt'
data1 = pd.read_csv(file_path, sep=';', low_memory=False, na_values='?', 
                   parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True)

# Check for missing values and create 'year' and 'year_month' columns for grouping
data1['year'] = data['datetime'].dt.year
data1['year_month'] = data['datetime'].dt.to_period('M')

# Calculate missing values per month and per year
missing_data_per_year = data1.isnull().groupby(data1['year']).sum()

# Plot missing values count per year for each feature
plt.figure(figsize=(14, 8))
missing_data_per_year.plot(kind='bar', stacked=True, figsize=(14, 8), cmap='viridis')
plt.title('Count of Missing Values per Year by Feature', fontsize=16)
plt.ylabel('Count of Missing Values', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
data.iloc[:, 1:] = imputer.fit_transform(data.iloc[:, 1:])
missing_values = data.isnull().sum()
print("Missing values handled - Mean imputation")

######################################
# Set the datetime column as the index
data.set_index('datetime', inplace=True)

# Resample data to daily mean
daily_data = data.resample('D').mean()
weekly_data = data.resample('W').mean()
monthly_data = data.resample('M').mean()
quarterly_data = data.resample('Q').mean()

# Visualize time series trends in the same subplot
plt.figure(figsize=(14, 7))
plt.plot(daily_data['Global_active_power'], label='Daily Global Active Power')
plt.plot(weekly_data['Global_active_power'], label='Weekly Global Active Power')
plt.plot(monthly_data['Global_active_power'], label='Monthly Global Active Power')
plt.plot(quarterly_data['Global_active_power'], label='Quarterly Global Active Power')
plt.title('Global Active Power: Daily, Weekly, Monthly, and Quarterly Trends')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()


# Analyze distribution of power consumption
plt.figure(figsize=(10, 6))
sns.histplot(daily_data['Global_active_power'].dropna(), kde=True)
plt.title('Distribution of Global Active Power')
plt.xlabel('Global Active Power (kilowatts)')
plt.ylabel('Frequency')
plt.show()

# Analyze correlation between variables
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='Purples', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Identify and handle outliers using a boxplot
plt.figure(figsize=(14, 7))
plt.boxplot(daily_data['Global_active_power'].dropna(), vert=False)
plt.title('Boxplot of Global Active Power')
plt.xlabel('Global Active Power (kilowatts)')
plt.show()

# Handling outliers by capping
q1 = daily_data['Global_active_power'].quantile(0.25)
q3 = daily_data['Global_active_power'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
daily_data['Global_active_power'] = np.where(daily_data['Global_active_power'] > upper_bound, upper_bound,
                                             np.where(daily_data['Global_active_power'] < lower_bound, lower_bound,
                                                      daily_data['Global_active_power']))

# Visualize the capped data
plt.figure(figsize=(14, 7))
plt.boxplot(daily_data['Global_active_power'].dropna(), vert=False)
plt.title('Boxplot of Capped Global Active Power (After Handling Outliers)')
plt.xlabel('Global Active Power (kilowatts)')
plt.show()

# Additional Visualizations for Yearly Analysis
# Extract year from datetime
daily_data['Year'] = daily_data.index.year

# Visualize yearly trends
plt.figure(figsize=(14, 7))
for year in daily_data['Year'].unique():
    yearly_data = daily_data[daily_data['Year'] == year]
    plt.plot(yearly_data.index, yearly_data['Global_active_power'], label=str(year))
plt.title('Yearly Global Active Power Trends')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Boxplot for each year
plt.figure(figsize=(14, 7))
sns.boxplot(x='Year', y='Global_active_power', data=daily_data)
plt.title('Yearly Boxplot of Global Active Power')
plt.xlabel('Year')
plt.ylabel('Global Active Power (kilowatts)')
plt.show()

# Monthly average consumption per year
monthly_data = daily_data.resample('M').mean()
monthly_data['Year'] = monthly_data.index.year
monthly_data['Month'] = monthly_data.index.month

plt.figure(figsize=(14, 7))
sns.lineplot(x='Month', y='Global_active_power', hue='Year', data=monthly_data, palette='tab10')
plt.title('Monthly Average Global Active Power per Year')
plt.xlabel('Month')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend(title='Year')
plt.show()

# Heatmap of monthly averages
monthly_avg = monthly_data.pivot_table(index='Month', columns='Year', values='Global_active_power')
plt.figure(figsize=(14, 7))
sns.heatmap(monthly_avg, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Heatmap of Monthly Average Global Active Power')
plt.xlabel('Year')
plt.ylabel('Month')
plt.show()

# %% 
# Question 3: Implement a linear regression model
print("ANSWER #3")
# Feature Engineering: Create lag features for 2 weeks (14 days)
for i in range(1, 15):
    daily_data[f'lag_{i}'] = daily_data['Global_active_power'].shift(i)

# Drop rows with NaN values due to shifting
daily_data.dropna(inplace=True)

# Define features and target variable
features = [f'lag_{i}' for i in range(1, 15)]
target = 'Global_active_power'

# Split the data so that the test set is the last three months
train_data = daily_data[:-90]
test_data = daily_data[-90:]

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

# Train the model and make predictions
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Question 4: Evaluate the linear regression model
print("ANSWER #4")
# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results")
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R²) value: {r2:.4f}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_test, label='Actual')
plt.plot(test_data.index, y_pred, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Prepare the performance comparison dataframe
performance_comparison = pd.DataFrame({
    'Model': ['Linear Regression'],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R²': [r2]
})

# Apply the highlight_max and highlight_min styles for performance_comparison
styled_df = (performance_comparison.style
                 .highlight_max(subset=['R²'], color='green')
                 .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='red'))

# Display the styled DataFrame
styled_df
# %% 
# Question 5: Implement a Recurrent Neural Network (RNN)
print("ANSWER #5")
# Preprocess data for RNN input
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_data[['Global_active_power']])

# Prepare the data for RNN input
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 14  # Sequence length set to 14 days
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split the data into training and testing sets (last 3 months as test set)
split = len(X) - 90
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = y_train.shape[1]
num_epochs = 100
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
model_rnn = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_rnn.parameters(), lr=learning_rate)

# Train the RNN model
rnn_history = []
for epoch in range(num_epochs):
    model_rnn.train()
    outputs = model_rnn(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    rnn_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot training loss for RNN
plt.figure(figsize=(10, 6))
plt.plot(rnn_history, label='Training Loss')
plt.title('Training Loss over Epochs (RNN)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the RNN model
model_rnn.eval()
with torch.no_grad():
    y_pred_train = model_rnn(X_train)
    y_pred_test = model_rnn(X_test)

# Inverse transform the predictions and true values to original scale
y_pred_train = scaler.inverse_transform(y_pred_train.numpy())
y_train = scaler.inverse_transform(y_train.numpy())
y_pred_test = scaler.inverse_transform(y_pred_test.numpy())
y_test = scaler.inverse_transform(y_test.numpy())

# Calculate performance metrics
mae_rnn = mean_absolute_error(y_test, y_pred_test)
mse_rnn = mean_squared_error(y_test, y_pred_test)
rmse_rnn = np.sqrt(mse_rnn)
r2_rnn = r2_score(y_test, y_pred_test)

print("RNN Results")
print(f'Mean Absolute Error (MAE): {mae_rnn:.4f}')
print(f'Mean Squared Error (MSE): {mse_rnn:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_rnn:.4f}')
print(f'R-squared (R²) value: {r2_rnn:.4f}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Actual')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_pred_test, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power using RNN')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Prepare the performance comparison dataframe
performance_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'RNN'],
    'MAE': [mae, mae_rnn],
    'MSE': [mse, mse_rnn],
    'RMSE': [rmse, rmse_rnn],
    'R²': [r2, r2_rnn]
})

# Apply the highlight_max and highlight_min styles for performance_comparison
styled_df = (performance_comparison.style
                 .highlight_max(subset=['R²'], color='green')
                 .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='red'))

# Display the styled DataFrame
styled_df
# %% 
# Question 6: Implement Long Short-Term Memory (LSTM)
print("ANSWER #6")
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
history = lstm_model.fit(X_train, y_train, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM
plot_training_history(history, "LSTM", "Resampled Data")

lstm_predictions = lstm_model.predict(X_test).flatten()

# Evaluate LSTM model
mae_lstm = mean_absolute_error(y_test, lstm_predictions)
mse_lstm = mean_squared_error(y_test, lstm_predictions)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_test, lstm_predictions)

print("LSTM Results")
print(f'Mean Absolute Error (MAE): {mae_lstm:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm:.4f}')
print(f'R-squared (R²) value: {r2_lstm:.4f}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Actual')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), lstm_predictions, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power using LSTM')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Prepare the performance comparison dataframe
performance_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'RNN', 'LSTM'],
    'MAE': [mae, mae_rnn, mae_lstm],
    'MSE': [mse, mse_rnn, mse_lstm],
    'RMSE': [rmse, rmse_rnn, rmse_lstm],
    'R²': [r2, r2_rnn, r2_lstm]
})

# Apply the highlight_max and highlight_min styles for performance_comparison
styled_df = (performance_comparison.style
                 .highlight_max(subset=['R²'], color='green')
                 .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='red'))

# Display the styled DataFrame
styled_df

# %% 
# Question 7: Implement LSTM with Attention
# Question 7: Implement LSTM with Attention
print("ANSWER #7")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1), a  # Return context vector and attention weights

# Define the LSTM with Attention model
inputs = tf.keras.Input(shape=(SEQ_LENGTH, 1))
lstm_out = LSTM(50, activation='relu', return_sequences=True)(inputs)
context_vector, attention_weights = AttentionLayer()(lstm_out)
outputs = Dense(1)(context_vector)

lstm_attention_model = tf.keras.Model(inputs=inputs, outputs=outputs)
lstm_attention_model.compile(optimizer='adam', loss='mse')
history = lstm_attention_model.fit(X_train, y_train, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM with Attention
plot_training_history(history, "LSTM with Attention", "Resampled Data")

lstm_attention_predictions = lstm_attention_model.predict(X_test).flatten()

# Evaluate LSTM with Attention model
mae_lstm_att = mean_absolute_error(y_test, lstm_attention_predictions)
mse_lstm_att = mean_squared_error(y_test, lstm_attention_predictions)
rmse_lstm_att = np.sqrt(mse_lstm_att)
r2_lstm_att = r2_score(y_test, lstm_attention_predictions)

print("LSTM with Attention Results")
print(f'Mean Absolute Error (MAE): {mae_lstm_att:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm_att:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm_att:.4f}')
print(f'R-squared (R²) value: {r2_lstm_att:.4f}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Actual')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), lstm_attention_predictions, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power using LSTM with Attention')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Model to extract attention weights
attention_model = tf.keras.Model(inputs=lstm_attention_model.input, 
                                 outputs=lstm_attention_model.layers[2].output[1])
attention_weights = attention_model.predict(X_test)

# Visualize attention weights
def plot_attention_weights(attention_weights, sample_idx):
    weights = attention_weights[sample_idx].flatten()
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights)
    plt.title('Attention Weights')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.show()

# Plot attention weights for a sample
plot_attention_weights(attention_weights, sample_idx=0)

# Prepare the performance comparison dataframe
performance_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'RNN', 'LSTM', 'LSTM with Attention'],
    'MAE': [mae, mae_rnn, mae_lstm, mae_lstm_att],
    'MSE': [mse, mse_rnn, mse_lstm, mse_lstm_att],
    'RMSE': [rmse, rmse_rnn, rmse_lstm, rmse_lstm_att],
    'R²': [r2, r2_rnn, r2_lstm, r2_lstm_att]
})

# Apply the highlight_max and highlight_min styles for performance_comparison
styled_df = (performance_comparison.style
                 .highlight_max(subset=['R²'], color='green')
                 .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='red'))

# Display the styled DataFrame
display(styled_df)

# %% 
# Question 8: Data Augmentation Experiment
print("ANSWER #8")
# Augment the data by modifying up to 10% of the dataset
augmented_df = daily_data.copy()
num_modifications = int(0.1 * len(daily_data))
random_indices = np.random.choice(daily_data.index, size=num_modifications, replace=False)
augmented_df.loc[random_indices, 'Global_active_power'] *= np.random.uniform(0.9, 1.1, size=num_modifications)

# Split augmented data
train_aug, test_aug = train_test_split(augmented_df, test_size=0.2, shuffle=False)
train_aug['lag1'] = train_aug['Global_active_power'].shift(1)
train_aug = train_aug.dropna()

# Define features and target variable
features = [f'lag_{i}' for i in range(1, 15)]
target = 'Global_active_power'

# Prepare features (consider lag variables, time-based features)
for i in range(1, 15):
    train_aug[f'lag_{i}'] = train_aug['Global_active_power'].shift(i)
train_aug = train_aug.dropna()

X_train_aug = train_aug[features]
y_train_aug = train_aug[target]

# Apply the same lag feature creation to the test set
for i in range(1, 15):
    test_aug[f'lag_{i}'] = test_aug['Global_active_power'].shift(i)
test_aug = test_aug.dropna()

X_test_aug = test_aug[features]
y_test_aug = test_aug[target]

# RNN with augmented data
scaler_aug = MinMaxScaler()
scaled_data_aug = scaler_aug.fit_transform(augmented_df[['Global_active_power']])

X_train_rnn_aug, y_train_rnn_aug = create_sequences(scaled_data_aug[:-len(test_aug)], SEQ_LENGTH)
X_test_rnn_aug, y_test_rnn_aug = create_sequences(scaled_data_aug[-len(test_aug)-SEQ_LENGTH:], SEQ_LENGTH)

X_train_rnn_aug = torch.tensor(X_train_rnn_aug, dtype=torch.float32)
y_train_rnn_aug = torch.tensor(y_train_rnn_aug, dtype=torch.float32)
X_test_rnn_aug = torch.tensor(X_test_rnn_aug, dtype=torch.float32)
y_test_rnn_aug = torch.tensor(y_test_rnn_aug, dtype=torch.float32)

# Initialize the RNN model for augmented data, loss function, and optimizer
model_rnn_aug = RNN(input_size, hidden_size, num_layers, output_size)
criterion_aug = nn.MSELoss()
optimizer_aug = torch.optim.Adam(model_rnn_aug.parameters(), lr=learning_rate)

# Train the RNN model with augmented data
rnn_history_aug = []
for epoch in range(num_epochs):
    model_rnn_aug.train()
    outputs = model_rnn_aug(X_train_rnn_aug)
    optimizer_aug.zero_grad()
    loss = criterion_aug(outputs, y_train_rnn_aug)
    loss.backward()
    optimizer_aug.step()
    rnn_history_aug.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot training loss for RNN with augmented data
plt.figure(figsize=(10, 6))
plt.plot(rnn_history_aug, label='Training Loss')
plt.title('Training Loss over Epochs (RNN with Augmented Data)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the RNN model with augmented data
model_rnn_aug.eval()
with torch.no_grad():
    rnn_predictions_aug = model_rnn_aug(X_test_rnn_aug).numpy().flatten()

# Inverse transform the predictions
rnn_predictions_aug = scaler_aug.inverse_transform(rnn_predictions_aug.reshape(-1, 1)).flatten()

mae_rnn_aug = mean_absolute_error(y_test_aug, rnn_predictions_aug)
mse_rnn_aug = mean_squared_error(y_test_aug, rnn_predictions_aug)
rmse_rnn_aug = np.sqrt(mse_rnn_aug)
r2_rnn_aug = r2_score(y_test_aug, rnn_predictions_aug)

print("Augmented RNN Results")
print(f'Mean Absolute Error (MAE): {mae_rnn_aug:.4f}')
print(f'Mean Squared Error (MSE): {mse_rnn_aug:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_rnn_aug:.4f}')
print(f'R-squared (R²) value: {r2_rnn_aug:.4f}')

# Plot the results for RNN
plt.figure(figsize=(14, 7))
plt.plot(test_aug.index, y_test_aug, label='Actual')
plt.plot(test_aug.index, rnn_predictions_aug, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (RNN) - Augmented Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# LSTM with augmented data
history = lstm_model.fit(X_train_rnn_aug, y_train_rnn_aug, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM with augmented data
plot_training_history(history, "LSTM", "Augmented Data")

lstm_predictions_aug = lstm_model.predict(X_test_rnn_aug).flatten()

# Inverse transform the predictions
lstm_predictions_aug = scaler_aug.inverse_transform(lstm_predictions_aug.reshape(-1, 1)).flatten()

mae_lstm_aug = mean_absolute_error(y_test_aug, lstm_predictions_aug)
mse_lstm_aug = mean_squared_error(y_test_aug, lstm_predictions_aug)
rmse_lstm_aug = np.sqrt(mse_lstm_aug)
r2_lstm_aug = r2_score(y_test_aug, lstm_predictions_aug)

print("Augmented LSTM Results")
print(f'Mean Absolute Error (MAE): {mae_lstm_aug:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm_aug:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm_aug:.4f}')
print(f'R-squared (R²) value: {r2_lstm_aug:.4f}')


# Plot the results for LSTM
plt.figure(figsize=(14, 7))
plt.plot(test_aug.index, y_test_aug, label='Actual')
plt.plot(test_aug.index, lstm_predictions_aug, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (LSTM) - Augmented Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# LSTM with Attention and augmented data
history = lstm_attention_model.fit(X_train_rnn_aug, y_train_rnn_aug, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM with Attention and augmented data
plot_training_history(history, "LSTM with Attention", "Augmented Data")

lstm_attention_predictions_aug = lstm_attention_model.predict(X_test_rnn_aug).flatten()

# Inverse transform the predictions
lstm_attention_predictions_aug = scaler_aug.inverse_transform(lstm_attention_predictions_aug.reshape(-1, 1)).flatten()

mae_lstm_att_aug = mean_absolute_error(y_test_aug, lstm_attention_predictions_aug)
mse_lstm_att_aug = mean_squared_error(y_test_aug, lstm_attention_predictions_aug)
rmse_lstm_att_aug = np.sqrt(mse_lstm_att_aug)
r2_lstm_att_aug = r2_score(y_test_aug, lstm_attention_predictions_aug)

print("Augmented LSTM with Attention Results")
print(f'Mean Absolute Error (MAE): {mae_lstm_att_aug:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm_att_aug:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm_att_aug:.4f}')
print(f'R-squared (R²) value: {r2_lstm_att_aug:.4f}')


# Plot the results for LSTM with Attention
plt.figure(figsize=(14, 7))
plt.plot(test_aug.index, y_test_aug, label='Actual')
plt.plot(test_aug.index, lstm_attention_predictions_aug, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (LSTM with Attention) - Augmented Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Prepare the performance comparison dataframe for augmented data
performance_comparison_aug = pd.DataFrame({
    'Model': ['RNN', 'LSTM', 'LSTM with Attention'],
    'MAE': [mae_rnn_aug, mae_lstm_aug, mae_lstm_att_aug],
    'MSE': [mse_rnn_aug, mse_lstm_aug, mse_lstm_att_aug],
    'RMSE': [rmse_rnn_aug, rmse_lstm_aug, rmse_lstm_att_aug],
    'R²': [r2_rnn_aug, r2_lstm_aug, r2_lstm_att_aug]
})

# Apply the highlight_max and highlight_min styles for performance_comparison_aug
styled_df_aug = (performance_comparison_aug.style
                 .highlight_max(subset=['R²'], color='green')
                 .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='red'))

# Display the styled DataFrame
display(styled_df_aug)

# Plot combined results
plt.figure(figsize=(14, 7))
plt.plot(test_aug.index, y_test_aug, label='Actual')
plt.plot(test_aug.index, rnn_predictions_aug, label='RNN', linestyle='--')
plt.plot(test_aug.index, lstm_predictions_aug, label='LSTM', linestyle='--')
plt.plot(test_aug.index, lstm_attention_predictions_aug, label='LSTM with Attention', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (All Models) - Augmented Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# %%
# Question 9: Data Reduction Experiment
print("ANSWER #9")
# Reduce the data by removing up to 10% of the dataset
reduced_df = daily_data.drop(daily_data.sample(frac=0.1, random_state=1).index)

# Split reduced data
train_red, test_red = train_test_split(reduced_df, test_size=0.2, shuffle=False)
train_red['lag1'] = train_red['Global_active_power'].shift(1)
train_red = train_red.dropna()

# Define features and target variable
features = [f'lag_{i}' for i in range(1, 15)]
target = 'Global_active_power'

# Prepare features (consider lag variables, time-based features)
for i in range(1, 15):
    train_red[f'lag_{i}'] = train_red['Global_active_power'].shift(i)
train_red = train_red.dropna()

X_train_red = train_red[features]
y_train_red = train_red[target]

# Apply the same lag feature creation to the test set
for i in range(1, 15):
    test_red[f'lag_{i}'] = test_red['Global_active_power'].shift(i)
test_red = test_red.dropna()

X_test_red = test_red[features]
y_test_red = test_red[target]

# RNN with reduced data
scaler_red = MinMaxScaler()
scaled_data_red = scaler_red.fit_transform(reduced_df[['Global_active_power']])

X_train_rnn_red, y_train_rnn_red = create_sequences(scaled_data_red[:-len(test_red)], SEQ_LENGTH)
X_test_rnn_red, y_test_rnn_red = create_sequences(scaled_data_red[-len(test_red)-SEQ_LENGTH:], SEQ_LENGTH)

X_train_rnn_red = torch.tensor(X_train_rnn_red, dtype=torch.float32)
y_train_rnn_red = torch.tensor(y_train_rnn_red, dtype=torch.float32)
X_test_rnn_red = torch.tensor(X_test_rnn_red, dtype=torch.float32)
y_test_rnn_red = torch.tensor(y_test_rnn_red, dtype=torch.float32)

# Initialize the RNN model for reduced data, loss function, and optimizer
model_rnn_red = RNN(input_size, hidden_size, num_layers, output_size)
criterion_red = nn.MSELoss()
optimizer_red = torch.optim.Adam(model_rnn_red.parameters(), lr=learning_rate)

# Train the RNN model with reduced data
rnn_history_red = []
for epoch in range(num_epochs):
    model_rnn_red.train()
    outputs = model_rnn_red(X_train_rnn_red)
    optimizer_red.zero_grad()
    loss = criterion_red(outputs, y_train_rnn_red)
    loss.backward()
    optimizer_red.step()
    rnn_history_red.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot training loss for RNN with reduced data
plt.figure(figsize=(10, 6))
plt.plot(rnn_history_red, label='Training Loss')
plt.title('Training Loss over Epochs (RNN with Reduced Data)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the RNN model with reduced data
model_rnn_red.eval()
with torch.no_grad():
    rnn_predictions_red = model_rnn_red(X_test_rnn_red).numpy().flatten()

# Inverse transform the predictions
rnn_predictions_red = scaler_red.inverse_transform(rnn_predictions_red.reshape(-1, 1)).flatten()

mae_rnn_red = mean_absolute_error(y_test_red, rnn_predictions_red)
mse_rnn_red = mean_squared_error(y_test_red, rnn_predictions_red)
rmse_rnn_red = np.sqrt(mse_rnn_red)
r2_rnn_red = r2_score(y_test_red, rnn_predictions_red)


print("Reduced RNN Results")
print(f'Mean Absolute Error (MAE): {mae_rnn_red:.4f}')
print(f'Mean Squared Error (MSE): {mse_rnn_red:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_rnn_red:.4f}')
print(f'R-squared (R²) value: {r2_rnn_red:.4f}')


# Plot the results for RNN
plt.figure(figsize=(14, 7))
plt.plot(test_red.index, y_test_red, label='Actual')
plt.plot(test_red.index, rnn_predictions_red, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (RNN) - Reduced Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# LSTM with reduced data
history = lstm_model.fit(X_train_rnn_red, y_train_rnn_red, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM with reduced data
plot_training_history(history, "LSTM", "Reduced Data")

lstm_predictions_red = lstm_model.predict(X_test_rnn_red).flatten()

# Inverse transform the predictions
lstm_predictions_red = scaler_red.inverse_transform(lstm_predictions_red.reshape(-1, 1)).flatten()

mae_lstm_red = mean_absolute_error(y_test_red, lstm_predictions_red)
mse_lstm_red = mean_squared_error(y_test_red, lstm_predictions_red)
rmse_lstm_red = np.sqrt(mse_lstm_red)
r2_lstm_red = r2_score(y_test_red, lstm_predictions_red)

print("Reduced LSTM Results")
print(f'Mean Absolute Error (MAE): {mae_lstm_red:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm_red:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm_red:.4f}')
print(f'R-squared (R²) value: {r2_lstm_red:.4f}')


# Plot the results for LSTM
plt.figure(figsize=(14, 7))
plt.plot(test_red.index, y_test_red, label='Actual')
plt.plot(test_red.index, lstm_predictions_red, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (LSTM) - Reduced Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# LSTM with Attention and reduced data
history = lstm_attention_model.fit(X_train_rnn_red, y_train_rnn_red, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM with Attention and reduced data
plot_training_history(history, "LSTM with Attention", "Reduced Data")

lstm_attention_predictions_red = lstm_attention_model.predict(X_test_rnn_red).flatten()

# Inverse transform the predictions
lstm_attention_predictions_red = scaler_red.inverse_transform(lstm_attention_predictions_red.reshape(-1, 1)).flatten()

mae_lstm_att_red = mean_absolute_error(y_test_red, lstm_attention_predictions_red)
mse_lstm_att_red = mean_squared_error(y_test_red, lstm_attention_predictions_red)
rmse_lstm_att_red = np.sqrt(mse_lstm_att_red)
r2_lstm_att_red = r2_score(y_test_red, lstm_attention_predictions_red)

print("Reduced LSTM with Attention Results")
print(f'Mean Absolute Error (MAE): {mae_lstm_att_red:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm_att_red:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm_att_red:.4f}')
print(f'R-squared (R²) value: {r2_lstm_att_red:.4f}')


# Plot the results for LSTM with Attention
plt.figure(figsize=(14, 7))
plt.plot(test_red.index, y_test_red, label='Actual')
plt.plot(test_red.index, lstm_attention_predictions_red, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (LSTM with Attention) - Reduced Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Prepare the performance comparison dataframe for reduced data
performance_comparison_red = pd.DataFrame({
    'Model': ['RNN', 'LSTM', 'LSTM with Attention'],
    'MAE': [mae_rnn_red, mae_lstm_red, mae_lstm_att_red],
    'MSE': [mse_rnn_red, mse_lstm_red, mse_lstm_att_red],
    'RMSE': [rmse_rnn_red, rmse_lstm_red, rmse_lstm_att_red],
    'R²': [r2_rnn_red, r2_lstm_red, r2_lstm_att_red]
})

# Apply the highlight_max and highlight_min styles for performance_comparison_red
styled_df_red = (performance_comparison_red.style
                 .highlight_max(subset=['R²'], color='green')
                 .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='red'))

# Display the styled DataFrame
display(styled_df_red)

# Plot combined results
plt.figure(figsize=(14, 7))
plt.plot(test_red.index, y_test_red, label='Actual')
plt.plot(test_red.index, rnn_predictions_red, label='RNN', linestyle='--')
plt.plot(test_red.index, lstm_predictions_red, label='LSTM', linestyle='--')
plt.plot(test_red.index, lstm_attention_predictions_red, label='LSTM with Attention', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (All Models) - Reduced Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()
# %% 
# Question 10: Data Resolution Experiment
print("ANSWER #10")
# Resample the data to 2-minute intervals
resampled_df = data.resample('2T').mean().dropna()

daily_data_2m = resampled_df.resample('D').mean()

# Feature Engineering: Create lag features for 2 weeks (14 days)
for i in range(1, 15):
    daily_data_2m[f'lag_{i}'] = daily_data_2m['Global_active_power'].shift(i)

# Drop rows with NaN values due to shifting
daily_data_2m.dropna(inplace=True)

# Define features and target variable
features = [f'lag_{i}' for i in range(1, 15)]
target = 'Global_active_power'

# Split the data into training and testing sets
train_res, test_res = train_test_split(daily_data_2m, test_size=0.2, shuffle=False)

X_train_res, y_train_res = train_res[features], train_res[target]
X_test_res, y_test_res = test_res[features], test_res[target]

# RNN with resampled data
scaler_res = MinMaxScaler()
scaled_data_res = scaler_res.fit_transform(daily_data_2m[['Global_active_power']])

X_train_rnn_res, y_train_rnn_res = create_sequences(scaled_data_res[:-len(test_res)], SEQ_LENGTH)
X_test_rnn_res, y_test_rnn_res = create_sequences(scaled_data_res[-len(test_res)-SEQ_LENGTH:], SEQ_LENGTH)

X_train_rnn_res = torch.tensor(X_train_rnn_res, dtype=torch.float32).reshape(-1, SEQ_LENGTH, 1)
y_train_rnn_res = torch.tensor(y_train_rnn_res, dtype=torch.float32)
X_test_rnn_res = torch.tensor(X_test_rnn_res, dtype=torch.float32).reshape(-1, SEQ_LENGTH, 1)
y_test_rnn_res = torch.tensor(y_test_rnn_res, dtype=torch.float32)

rnn_model_res = RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
criterion_res = nn.MSELoss()
optimizer_res = torch.optim.Adam(rnn_model_res.parameters(), lr=learning_rate)

# Train the RNN model with resampled data
rnn_history_res = []
for epoch in range(num_epochs):
    rnn_model_res.train()
    outputs = rnn_model_res(X_train_rnn_res)
    optimizer_res.zero_grad()
    loss = criterion_res(outputs, y_train_rnn_res)
    loss.backward()
    optimizer_res.step()
    rnn_history_res.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot training loss for RNN with resampled data
plt.figure(figsize=(10, 6))
plt.plot(rnn_history_res, label='Training Loss')
plt.title('Training Loss over Epochs (RNN with Resampled Data)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the RNN model with resampled data
rnn_model_res.eval()
with torch.no_grad():
    rnn_predictions_res = rnn_model_res(X_test_rnn_res).numpy().flatten()

mae_rnn_res = mean_absolute_error(y_test_rnn_res.numpy(), rnn_predictions_res)
mse_rnn_res = mean_squared_error(y_test_rnn_res.numpy(), rnn_predictions_res)
rmse_rnn_res = np.sqrt(mse_rnn_res)
r2_rnn_res = r2_score(y_test_rnn_res.numpy(), rnn_predictions_res)

print("Resampled RNN Results")
print(f'Mean Absolute Error (MAE): {mae_rnn_res:.4f}')
print(f'Mean Squared Error (MSE): {mse_rnn_res:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_rnn_res:.4f}')
print(f'R-squared (R²) value: {r2_rnn_res:.4f}')



# Plot the results for RNN
plt.figure(figsize=(14, 7))
plt.plot(test_res.index[-len(y_test_rnn_res):], y_test_rnn_res.numpy(), label='Actual')
plt.plot(test_res.index[-len(y_test_rnn_res):], rnn_predictions_res, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (RNN) - Resampled Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# LSTM with resampled data
history = lstm_model.fit(X_train_rnn_res, y_train_rnn_res, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM with resampled data
plot_training_history(history, "LSTM", "Resampled Data")

lstm_predictions_res = lstm_model.predict(X_test_rnn_res).flatten()

mae_lstm_res = mean_absolute_error(y_test_rnn_res.numpy(), lstm_predictions_res)
mse_lstm_res = mean_squared_error(y_test_rnn_res.numpy(), lstm_predictions_res)
rmse_lstm_res = np.sqrt(mse_lstm_res)
r2_lstm_res = r2_score(y_test_rnn_res.numpy(), lstm_predictions_res)

print("Resampled LSTM Results")
print(f'Mean Absolute Error (MAE): {mae_lstm_res:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm_res:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm_res:.4f}')
print(f'R-squared (R²) value: {r2_lstm_res:.4f}')


# Plot the results for LSTM
plt.figure(figsize=(14, 7))
plt.plot(test_res.index[-len(y_test_rnn_res):], y_test_rnn_res.numpy(), label='Actual')
plt.plot(test_res.index[-len(y_test_rnn_res):], lstm_predictions_res, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (LSTM) - Resampled Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# LSTM with Attention and resampled data
history = lstm_attention_model.fit(X_train_rnn_res, y_train_rnn_res, epochs=num_epochs, batch_size=32)

# Plot training loss for LSTM with Attention and resampled data
plot_training_history(history, "LSTM with Attention", "Resampled Data")

lstm_attention_predictions_res = lstm_attention_model.predict(X_test_rnn_res).flatten()

mae_lstm_att_res = mean_absolute_error(y_test_rnn_res.numpy(), lstm_attention_predictions_res)
mse_lstm_att_res = mean_squared_error(y_test_rnn_res.numpy(), lstm_attention_predictions_res)
rmse_lstm_att_res = np.sqrt(mse_lstm_att_res)
r2_lstm_att_res = r2_score(y_test_rnn_res.numpy(), lstm_attention_predictions_res)

print("Resampled LSTM with Attention Results")
print(f'Mean Absolute Error (MAE): {mae_lstm_att_res:.4f}')
print(f'Mean Squared Error (MSE): {mse_lstm_att_res:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm_att_res:.4f}')
print(f'R-squared (R²) value: {r2_lstm_att_res:.4f}')


# Plot the results for LSTM with Attention
plt.figure(figsize=(14, 7))
plt.plot(test_res.index[-len(y_test_rnn_res):], y_test_rnn_res.numpy(), label='Actual')
plt.plot(test_res.index[-len(y_test_rnn_res):], lstm_attention_predictions_res, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (LSTM with Attention) - Resampled Data')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

# Prepare the performance comparison dataframe for resampled data
performance_comparison_res = pd.DataFrame({
    'Model': ['RNN', 'LSTM', 'LSTM with Attention'],
    'MAE': [mae_rnn_res, mae_lstm_res, mae_lstm_att_res],
    'MSE': [mse_rnn_res, mse_lstm_res, mse_lstm_att_res],
    'RMSE': [rmse_rnn_res, rmse_lstm_res, rmse_lstm_att_res],
    'R²': [r2_rnn_res, r2_lstm_res, r2_lstm_att_res]
})

# Apply the highlight_max and highlight_min styles for performance_comparison_res
styled_df_res = (performance_comparison_res.style
                 .highlight_max(subset=['R²'], color='green')
                 .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='red'))

# Display the styled DataFrame
display(styled_df_res)

# Combined plot for actual and predicted values from all models
plt.figure(figsize=(14, 7))
plt.plot(test_res.index[-len(y_test_rnn_res):], y_test_rnn_res.numpy(), label='Actual')
plt.plot(test_res.index[-len(y_test_rnn_res):], rnn_predictions_res, label='RNN', linestyle='--')
plt.plot(test_res.index[-len(y_test_rnn_res):], lstm_predictions_res, label='LSTM', linestyle='--')
plt.plot(test_res.index[-len(y_test_rnn_res):], lstm_attention_predictions_res, label='LSTM with Attention', linestyle='--')
plt.title('Actual vs Predicted Global Active Power (All Models - Resampled Data)')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()

