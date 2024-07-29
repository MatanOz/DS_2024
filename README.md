
# Power Consumption Prediction Project

This project aims to analyze and predict power consumption using various machine learning models. The dataset used is the "Individual household electric power consumption" dataset from the UCI Machine Learning Repository. The project includes exploratory data analysis (EDA), and implementing and evaluating multiple models: Linear Regression, Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and LSTM with Attention. Additionally, experiments on data augmentation, reduction, and resolution were conducted to observe their impacts on model performance.

## Project Structure

1. **Introduction**
2. **Data Description**
3. **Exploratory Data Analysis (EDA)**
4. **Linear Regression Model**
5. **Recurrent Neural Network (RNN) Model**
6. **Long Short-Term Memory (LSTM) Model**
7. **LSTM with Attention Layer**
8. **Data Augmentation Experiment**
9. **Data Reduction Experiment**
10. **Data Resolution Experiment**
11. **Overall Conclusions**

## Introduction

The goal is to predict power consumption and explore how different models and data preprocessing techniques affect prediction accuracy.

## Data Description

The dataset includes measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost four years. Key features include:

- Date
- Time
- Global_active_power
- Global_reactive_power
- Voltage
- Global_intensity
- Sub_metering_1, Sub_metering_2, Sub_metering_3

## Exploratory Data Analysis (EDA)

EDA involves visualizing time series trends, checking for seasonality and cyclical patterns, analyzing the distribution of power consumption, and identifying and handling missing values and outliers.

## Model Implementations

### Linear Regression Model

A basic model to establish a performance baseline.

### Recurrent Neural Network (RNN) Model

An RNN model was trained to capture temporal dependencies in the data.

### Long Short-Term Memory (LSTM) Model

An LSTM model was implemented to better capture long-term dependencies.

### LSTM with Attention Layer

An enhanced LSTM model with an Attention layer to focus on relevant parts of the data.

## Experiments

### Data Augmentation Experiment

Data was augmented by modifying up to 10% of the dataset. The LSTM model showed the most improvement from data augmentation.

### Data Reduction Experiment

10% of the data was randomly removed to assess model robustness. The RNN model was the most robust to data reduction.

### Data Resolution Experiment

The data was resampled to 2-minute intervals. All models benefited, with the LSTM with Attention model performing the best.

## Overall Conclusions

- **Linear Regression**: Performed the worst, unable to capture temporal dependencies.
- **RNN**: Showed significant improvement over Linear Regression and was robust to data reduction.
- **LSTM**: Benefited from data augmentation and higher resolution data, sensitive to data reduction.
- **LSTM with Attention**: Best overall performance with higher resolution data but mixed results with augmentation and reduction.

## Final Thoughts

The RNN and LSTM models outperformed Linear Regression. Data augmentation and higher resolution data improved model performance, especially for complex models like LSTM and LSTM with Attention. Future work could involve further tuning and exploring additional preprocessing techniques.

## How to Run

1. Install the required libraries from `requirements.txt`.
2. Load the dataset from the UCI Machine Learning Repository.
3. Follow the project structure to replicate the analysis and experiments.

## Requirements

- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch (for RNN implementation)
- TensorFlow/Keras (for LSTM and LSTM with Attention implementation)

