import pandas as pd
import numpy as np
from astral.sun import sun
from astral import LocationInfo
import datetime

tum_lat  = 48.267410
tum_long = 11.668265
time_abs_start = "2024-08-24T00:00:00Z"

class SineFitting():

    def __init__(self, 
                 date:datetime.datetime    = time_abs_start,
                 latitude:str              = tum_lat, 
                 longitude:str             = tum_long,
                 previous_df:pd.DataFrame  = pd.DataFrame,
                 num_frequencies:int       = 5, 
                 scaling:int               = 5
                 ) -> None:

        self.sunrise_min = float()
        self.sunrise_max = float()
        self.df = pd.DataFrame
        self.sine_modeled = np.array
        self.date = date

        self.set_latitude(latitude)
        self.set_longitude(longitude)
        self._calcSunValues()

        self.set_previous_day(previous_df)

        self.num_frequencies = num_frequencies
        self.scaling = scaling

        pass

    def __str__(self) -> str:
        return str(self.df)
    
    def set_latitude(self, lat):
        self.latitude = lat

    def set_longitude(self, long):
        self.longitude = long

    def _calcSunValues(self):
        city = LocationInfo(latitude=self.latitude, longitude=self.longitude)
        dtime = datetime.datetime.strptime(self.date, "%Y-%m-%dT%H:%M:%SZ")

        s = sun(city.observer, date=datetime.date(dtime.year, dtime.month, dtime.day))
        sunrise = s["sunrise"]
        sunset  = s["sunset"]

        self.sunrise_min = sunrise.hour + (sunrise.minute/60) + 1.15
        self.sunrise_max = sunset.hour + (sunset.minute/60) - 0.6

    def getSunrise(self) -> float:
        return self.sunrise_min
    
    def getSunset(self) -> float:
        return self.sunrise_max

    def set_previous_day(self, data_previous) -> None:
        self.df = data_previous

    def signal_fft(self, signal, num_frequencies):
        N = len(signal)  # Length of the input signal
        x_vals = np.linspace(0, 1, N)  # Input domain in [0, 1]

        # Step 1: FFT on the signal
        X = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N)  # Frequency bins

        # Step 2: Get magnitudes and select the top k frequencies
        magnitudes = np.abs(X)
        indices = np.argsort(magnitudes)[-num_frequencies:]  # Indices of the top k frequencies

        # Extract the important frequencies, amplitudes, and phases
        top_freqs = freqs[indices]  # Frequencies of the top k components
        top_amplitudes = np.abs(X[indices])  # Amplitudes of the top k components
        top_phases = np.angle(X[indices])  # Phases of the top k components
        return top_freqs, top_amplitudes, top_phases
    
    def reconstruction_func(self, x, top_freqs, top_amplitudes, top_phases, signal_length):
        # Ensure x is a numpy array for vectorized operations
        x = np.asarray(x)

        # Initialize the reconstructed signal
        reconstructed = np.zeros_like(x, dtype=float)

        print(f"top_freqs: {top_freqs}")
        print(f"top_amplitudes: {top_amplitudes}")
        print(f"top_phases: {top_phases}")

        # Rebuild the signal using the top k frequencies
        for freq, amp, phase in zip(top_freqs, top_amplitudes, top_phases):
            # Construct each component: A * cos(2π * f * x * N + phase)
            reconstructed += amp * np.cos(2 * np.pi * freq * x * signal_length + phase)
        # freq = max(top_freqs)
        # amp = max(top_amplitudes)
        # phase = max(top_phases)
        # reconstructed += amp * np.cos(2 * np.pi * freq * x * signal_length + phase)

        return reconstructed # * self.scaling
    
    def basis_function(self, t_min, sunrise_min, sunset_min, top_freqs, top_amplitudes, top_phases, signal_length):

        t_wave_input = (t_min - sunrise_min) / (sunset_min - sunrise_min)
        reconstructed = self.reconstruction_func(t_wave_input, top_freqs, top_amplitudes, top_phases, signal_length) * self.scaling
        reconstructed[(t_min < sunrise_min) | (t_min > sunset_min)] = 0

        return reconstructed

    def calc_sine(self) -> np.array:
        # Fix parameters for inital fitting
        fixed_sunrise_min = 5.5
        fixed_sunset_min = 17.5

        # Calc acutal sunrise / sunset
        self._calcSunValues()

        length_of_signal = len(self.df.index)
        x = self.df.INV2.values
        x = x[int(fixed_sunrise_min * 60) : int(fixed_sunset_min * 60)] # cut only the sine-parts of the waveform
        top_freqs, top_amplitudes, top_phases = self.signal_fft(x / x.max(), self.num_frequencies)
        signal_length = x.shape[0]

        xi = np.linspace(0, 24 * 60, length_of_signal)
        self.sine_modeled = self.basis_function(xi, self.sunrise_min * 60, self.sunrise_max * 60, top_freqs, top_amplitudes, top_phases, signal_length=signal_length)

        return self.sine_modeled

    def calc_d(self) -> np.array:
        self.dModel = self.df.INV2.values - self.sine_modeled
        return self.dModel
    