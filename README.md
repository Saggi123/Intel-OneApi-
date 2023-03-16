# Intel-OneApi-

This code is a Python script that loads historical stock data for the Apple Inc. (AAPL) company from Yahoo Finance, preprocesses the data by normalizing it using MinMaxScaler, creates a time series data sequence with a specified sequence length, builds a deep learning model with multiple LSTM layers, trains the model on the training data, saves the trained model to a file, and finally, uses the trained model to predict the future stock prices of AAPL and plots the actual and predicted prices on a graph.

More specifically, the code does the following:

Imports the necessary libraries: os, pandas, numpy, matplotlib, tensorflow, keras, and sklearn.

Sets up the OneAPI environment.

Defines the ticker symbol for Apple Inc. (AAPL) and loads the historical stock data for this company from Yahoo Finance using OneAPI.

Preprocesses the data by converting the date column to a datetime format, sorting the data in ascending order by date, splitting the data into training and testing sets, and normalizing the data using MinMaxScaler.

Defines a function called create_sequences that creates a time series data sequence with a specified sequence length.

Defines the sequence length as 100 and uses the create_sequences function to create the training and testing data sequences.

Reshapes the training and testing data sequences to be compatible with the LSTM layers in the deep learning model.

Builds a deep learning model using Keras Sequential API with multiple LSTM layers and a dense layer.

Compiles the model with an Adam optimizer and mean squared error loss.

Trains the model on the training data using 50 epochs and a batch size of 32, and validates the model on the testing data.

Saves the trained model to a file called keras_model.h5.

Defines a function called predict that uses the trained model to predict the future stock prices of AAPL.

Uses the predict function to predict the future stock prices of AAPL and plots the actual and predicted prices on a graph using Matplotlib.

#Cloud # Artificial Intelligence
