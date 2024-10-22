import numpy as np
import matplotlib.pyplot as plt
import librosa


def add_noise(audio_data: np.ndarray, snr_db: int = 20):
    # Calculate the power of the original signal
    signal_power = np.mean(audio_data**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    # Generate white noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_data))
    # Add noise to the original signal
    y_noisy = audio_data + noise
