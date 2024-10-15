import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def divide_in_frames(audio_path: str) -> np.ndarray:
    # Read the audio data and sampling rate
    sampling_rate, audio_data = wavfile.read(audio_path)

    # Calculate the number of samples per 20 ms frame
    frame_duration_ms = 20
    samples_per_frame = int(sampling_rate * frame_duration_ms / 1000)

    # Reshape the audio data into frames
    num_frames = len(audio_data) // samples_per_frame
    frames = np.reshape(audio_data[:num_frames * samples_per_frame], (num_frames, samples_per_frame))

    return frames


def apply_hamming_window(frame: np.ndarray) -> np.ndarray:
    # Generate a Hamming window the same length as the frame
    hamming_window = np.hamming(len(frame))

    # Apply the Hamming window to the frame by multiplying element-wise
    windowed_frame = frame * hamming_window

    return windowed_frame


def create_spectrogram(frame: np.ndarray) -> np.ndarray:
    """
    Create a spectrogram from a given audio frame using the Short-Time Fourier Transform (STFT).

    Parameters:
    frame (np.ndarray): The input audio frame.

    Returns:
    np.ndarray: The magnitude spectrogram of the input frame.
    """
    # Perform the Fourier transform (FFT) on the frame
    fft_result = np.fft.fft(frame)
    
    # Compute the magnitude spectrum (absolute values of the complex FFT result)
    magnitude_spectrum = np.abs(fft_result)
    
    # Return the magnitude spectrum as the spectrogram of the frame
    return magnitude_spectrum


def save_spectrogram_as_image(spectrogram: np.ndarray, filename: str) -> None:
    """
    Save the spectrogram as an image using matplotlib.

    Parameters:
    spectrogram (np.ndarray): The magnitude spectrogram to save as an image.
    filename (str): The filename to save the image as.
    """
    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
    plt.plot(spectrogram)  # Plot the spectrogram (1D magnitude spectrum)
    plt.title('Spectrogram')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    
    # Save the figure as an image file
    plt.savefig(filename)
    plt.close()  # Close the figure to free up memory


def main():
    audio_path = "your_audio_file.wav"
    # Example usage:
    frame = divide_in_frames(audio_path)[0]
    windowed_frame = apply_hamming_window(frame)
    spectrogram = create_spectrogram(windowed_frame)

    # Save the spectrogram as an image
    save_spectrogram_as_image(spectrogram, 'spectrogram_image.png')

    print("Spectrogram saved as 'spectrogram_image.png'")