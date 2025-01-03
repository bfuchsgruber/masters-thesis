"""
Class for loading a LSTM prediction model and forecast the next value for (dT = 15 min)

Author:     Benedikt Fuchsgruber
Mail:       benedikt.fuchsgruber@tum.de
Project:    Master's Thesis (MPC for the ZEI DERs)

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from .convert_utc import utc_to_local
from telemetry import tprint

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

class Prediction ():

    def __init__(self,
                 model_path:str="",
                 name:str="",
                 previous_df:pd.DataFrame=pd.DataFrame(),
                 verbose=False
                 ):
        
        # Save constructor arguments to object attributes
        self.model_path = model_path
        self.name = name
        self.previous_df = previous_df
        self.verbose = verbose

        # INV1 flag
        self.is_inv1 = False
        self.inv1_factor = 0.9657842157842158
        if self.name == "INV1":
            self.is_inv1 = True

        self.scaling = 1.0
        self.offset = 0.0

        # Create a model from the set path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 1
        self.num_layers = 3  # Increased number of layers (3)
        self.hidden_size = 128  # Increased number of hidden units (128)
        self.output_size = 1
        self.dropout = 0.2  # Added dropout for regularization
        self.load_model()

        pass

    def load_model(self) -> None:
        self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.status("Model successfully loaded")

    def set_previous_df(self, df:pd.DataFrame) -> None:
        self.previous_df = df.copy(deep=True)

    def get_previous_df(self) -> pd.DataFrame:
        return self.previous_df
    
    def set_scaling(self, scaling:float) -> None:
        self.scaling = scaling

    def set_offset(self, offset:float) -> None:
        self.offset = offset

    def predict(self, steps=1, convert_utc_to_local=True, float_output=False, p_inv2=0.0, p_inv3=0.0):
        # First, check if workaround for INV1 is required
        if self.is_inv1:
            pred_power_inv1 = (p_inv2 + p_inv3) * self.inv1_factor
            pred_power_inv1_np = np.float32(pred_power_inv1)
            return pred_power_inv1_np


        self.steps = steps

        train_data = self.previous_df.copy(deep=True)

        # Change the power column index (named with 'Name') to a generic index (called 'POWER')
        train_data.rename(columns=lambda x: x.replace(self.name, 'POWER'), inplace=True) # Rename the specific power column 'INV2' to 'POWER'


        # tprint(train_data.head())
        dataset_train = train_data.POWER.values
        
        # Reshaping 1D to 2D array
        dataset_train = np.reshape(dataset_train, (-1, 1))
        # tprint(dataset_train.shape)

        scaler = MinMaxScaler(feature_range=(0, 1))
        # Scaling dataset
        scaled_train = scaler.fit_transform(dataset_train)
        # tprint(scaled_train[:5])

        # Create sequences and labels for training data
        sequence_length = 50  # Number of time steps to look back
        X_train, y_train = [], []
        for i in range(len(scaled_train) - sequence_length):
            X_train.append(scaled_train[i:i + sequence_length])
            y_train.append(scaled_train[i + sequence_length])  # Predicting the value right after the sequence
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        # tprint(X_train.shape, y_train.shape)

        num_forecast_steps = self.steps # (30)
        time_freq = f"{num_forecast_steps}min"
        sequence_to_plot = X_train.squeeze().cpu().numpy()
        historical_data = sequence_to_plot[-1]

        forecasted_values = []
        with torch.no_grad():
            for _ in range(num_forecast_steps):
                historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(self.device)
                predicted_value = self.model(historical_data_tensor).cpu().numpy()[0, 0]
                forecasted_values.append(predicted_value)
                historical_data = np.roll(historical_data, shift=-1)
                historical_data[-1] = predicted_value

        last_date = train_data.index[-1]
        self.future_dates = pd.date_range(start=last_date + pd.Timedelta(15, unit='m'), periods=num_forecast_steps, freq=time_freq) # add 1 min steps to the df
        self.forecasted = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1)).flatten()
        
        # Convert UTC prediction time to local timezone
        self.future_dates_local_time = []
        if convert_utc_to_local:
            for i in self.future_dates:
                future_dates_local_time = utc_to_local(pd.to_datetime(i))
                self.future_dates_local_time.append(future_dates_local_time)
            self.future_dates = self.future_dates_local_time

        # Adjust predicted values
        self.forecasted *= self.scaling
        self.forecasted += self.offset


        df_prediction = pd.DataFrame({"Date":self.future_dates, self.name:self.forecasted})
        df_prediction.set_index("Date", inplace=True)

        if float_output:
            return self.value_from_df(df_prediction)
        else:
            return df_prediction
        # return self.future_dates, self.forecasted

    def value_from_df (self, df) -> float:
        val = df.iloc[0,0]
        if val < 0:
            val = np.float32(0.0)
        return val

    def __str__(self):
        return f"LSTM Model:{self.name}, Path:{self.model_path}"
    

    def status(self, text:str) -> tprint:
        if self.verbose:
            tprint(text)

