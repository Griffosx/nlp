import numpy as np
import matplotlib.pyplot as plt
import librosa


def add_noise():
    y, sr = librosa.load('M001_01_001.wav', sr=None)
    # Calculate the power of the original signal
    signal_power = np.mean (y **2)
    snr_db = 20
    noise_power = signal_power / (10**(snr_db / 10))
    # Generate white noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    # Add noise to the original signal
    y_noisy = y + noise
