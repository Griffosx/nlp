from pathlib import Path
from typing import Tuple
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
from pesq import pesq
import os


# Constants
N_FFT = 1024
HOP_LENGHT = 256
N_MELS = 80


def compute_mel_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Compute the Mel spectrogram from an audio signal.

    Parameters:
        audio_data (np.ndarray): Audio time series.
        sample_rate (int): Sampling rate.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of Mel bands.

    Returns:
        mel_spectrogram (np.ndarray): Mel spectrogram.
    """
    # Compute STFT to get the complex spectrogram
    stft = librosa.stft(audio_data, n_fft=N_FFT, hop_length=HOP_LENGHT)
    # Compute the magnitude spectrogram
    magnitude_spectrogram = np.abs(stft)
    # Convert the amplitude spectrogram to power spectrogram
    power_spectrogram = magnitude_spectrogram**2
    # Compute the Mel spectrogram from the power spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        S=power_spectrogram,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGHT,
        n_mels=N_MELS,
    )
    return mel_spectrogram


def invert_mel_spectrogram(mel_spectrogram: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Invert a Mel spectrogram back to a magnitude spectrogram.

    Parameters:
        mel_spectrogram (np.ndarray): Mel spectrogram.
        sample_rate (int): Sampling rate.
        n_fft (int): Length of the FFT window.
        n_mels (int): Number of Mel bands.

    Returns:
        magnitude_spectrogram_approx (np.ndarray): Approximated magnitude spectrogram.
    """
    # Create Mel filter bank
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=N_FFT, n_mels=N_MELS)
    # Compute pseudo-inverse of the Mel filter bank
    inv_mel_basis = np.linalg.pinv(mel_basis)
    # Invert the Mel spectrogram to approximate the power spectrogram
    power_spectrogram_approx = np.dot(inv_mel_basis, mel_spectrogram)
    # Ensure all values are non-negative
    power_spectrogram_approx = np.maximum(0, power_spectrogram_approx)
    # Convert power spectrogram back to amplitude spectrogram
    magnitude_spectrogram_approx = np.sqrt(power_spectrogram_approx)
    return magnitude_spectrogram_approx


def reconstruct_waveform(magnitude_spectrogram: np.ndarray) -> np.ndarray:
    """
    Reconstruct a time-domain waveform from a magnitude spectrogram using the Griffin-Lim algorithm.

    Parameters:
        magnitude_spectrogram (np.ndarray): Approximated magnitude spectrogram.
        n_iter (int): Number of iterations for Griffin-Lim algorithm.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): Length of the FFT window.

    Returns:
        reconstructed_audio (np.ndarray): Reconstructed audio time series.
    """
    n_iter = 60
    # Use Griffin-Lim algorithm to estimate the phase and reconstruct the signal
    reconstructed_audio = librosa.griffinlim(
        magnitude_spectrogram, n_iter=n_iter, hop_length=HOP_LENGHT, win_length=N_FFT
    )
    return reconstructed_audio


def save_audio(audio_data: np.ndarray, sample_rate: int, filename: str) -> None:
    """
    Save an audio time series to a WAV file.

    Parameters:
        audio_data (np.ndarray): Audio time series.
        sample_rate (int): Sampling rate.
        output_file (str): Path to the output file.
    """
    output_file = f"tts/audio_tts_generated/{filename}.wav"
    sf.write(output_file, audio_data, sample_rate)
    return output_file


def save_mel_spectrogram_plot(
    mel_spectrogram: np.ndarray,
    sample_rate: int,
    filename: str,
) -> None:
    """
    Plot and save the Mel spectrogram.
    """
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(
        mel_spectrogram_db,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        hop_length=HOP_LENGHT,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.savefig(
        f"tts/mel_spectrograms/{filename}.png", bbox_inches="tight", pad_inches=0.1
    )
    plt.close()


def save_waveform_plot(
    audio_data: np.ndarray, sample_rate: int, filename: str, title: str = "Waveform"
) -> None:
    """
    Plot and save the waveform of the audio data.
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(f"tts/waveforms/{filename}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_waveforms(
    original_audio_data: np.ndarray,
    reconstructed_audio_data: np.ndarray,
    sample_rate: int,
    filename: str,
) -> None:
    """
    Plot and save the comparison of original and reconstructed waveforms.
    """
    # Save the original waveform plot
    save_waveform_plot(original_audio_data, sample_rate, filename)

    # Save the reconstructed waveform plot
    save_waveform_plot(
        reconstructed_audio_data,
        sample_rate,
        f"reconstructed_{filename}",
        title="Reconstructed Waveform",
    )

    # Save the comparison
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(original_audio_data, sr=sample_rate)
    plt.title("Original Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(reconstructed_audio_data, sr=sample_rate)
    plt.title("Reconstructed Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(
        f"tts/waveform_comparisons/{filename}.png", bbox_inches="tight", pad_inches=0.1
    )
    plt.close()


def save_f0_contour(
    filename: str,
    original_audio_data: np.ndarray,
    reconstructed_audio_data: np.ndarray,
    sample_rate: int,
    fmin: float = librosa.note_to_hz("C2"),
    fmax: float = librosa.note_to_hz("C7"),
    title: str = "F0 Contour Comparison",
) -> None:
    """
    Extract and plot the F0 contours of the original and reconstructed audio signals.

    Parameters:
        original_audio_data (np.ndarray): Original audio time series.
        reconstructed_audio_data (np.ndarray): Reconstructed audio time series.
        sample_rate (int): Sampling rate.
        fmin (float): Minimum frequency in Hz.
        fmax (float): Maximum frequency in Hz.
        title (str): Title of the plot.
    """
    img_filename = filename.split(".")[0]
    # Extract F0 contours
    f0_original, times = extract_f0(
        original_audio_data, sample_rate, fmin=fmin, fmax=fmax
    )
    f0_reconstructed, _ = extract_f0(
        reconstructed_audio_data, sample_rate, fmin=fmin, fmax=fmax
    )

    # Plot F0 contour comparison
    plt.figure(figsize=(14, 5))
    plt.plot(times, f0_original, label="Original F0")
    plt.plot(times, f0_reconstructed, label="Reconstructed F0", alpha=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    # Save with default bounding box and padding
    plt.savefig(
        f"tts/f0_contours/{img_filename}.png", bbox_inches="tight", pad_inches=0.1
    )
    plt.close()


def extract_f0(
    audio_data: np.ndarray,
    sample_rate: int,
    fmin: float = librosa.note_to_hz("C2"),
    fmax: float = librosa.note_to_hz("C7"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the fundamental frequency (F0) contour from an audio signal.

    Parameters:
        audio_data (np.ndarray): Audio time series.
        sample_rate (int): Sampling rate.
        fmin (float): Minimum frequency in Hz.
        fmax (float): Maximum frequency in Hz.

    Returns:
        f0 (np.ndarray): Array of fundamental frequencies.
        times (np.ndarray): Array of time stamps corresponding to f0.
    """
    # Use librosa.pyin to estimate F0
    f0, _, _ = librosa.pyin(audio_data, fmin=fmin, fmax=fmax)
    times = librosa.times_like(f0, sr=sample_rate, hop_length=512)
    return f0, times


def perform_pesq_evaluation(
    original_audio_path: str, generated_audio_path: str, sample_rate: int = 16000
) -> float:
    """
    Perform PESQ evaluation between the original and reconstructed audio files.

    Parameters:
        original_audio_path (str): Path to the original audio file.
        generated_audio_path (str): Path to the reconstructed audio file.
        sample_rate (int): Sampling rate for PESQ evaluation.

    Returns:
        pesq_score (float): PESQ score.
    """
    _original_sample_rate, original_audio = wavfile.read(original_audio_path)
    _generated_sample_rate, generated_audio = wavfile.read(generated_audio_path)

    # PESQ supports only sample rates of 8000 or 16000 Hz
    if sample_rate not in [8000, 16000]:
        raise ValueError("PESQ evaluation requires sample rate to be 8000 or 16000 Hz")

    pesq_score = pesq(sample_rate, original_audio, generated_audio, "wb")
    print(f"PESQ Score: {pesq_score}")
    return pesq_score


def tts_pipeline(original_audio_path: str) -> None:
    """
    Text-to-Speech processing pipeline.
    """
    filename = original_audio_path.split("/")[-1].split(".")[0]

    # Ensure that all output directories exist
    os.makedirs("tts/audio_tts_generated", exist_ok=True)
    os.makedirs("tts/mel_spectrograms", exist_ok=True)
    os.makedirs("tts/waveforms", exist_ok=True)
    os.makedirs("tts/waveform_comparisons", exist_ok=True)
    os.makedirs("tts/f0_contours", exist_ok=True)

    # Load the original audio
    original_audio_data, sample_rate = librosa.load(original_audio_path, sr=None)

    # Compute the Mel spectrogram
    mel_spectrogram = compute_mel_spectrogram(
        original_audio_data,
        sample_rate,
    )

    # Invert the Mel spectrogram back to a magnitude spectrogram
    magnitude_spectrogram_approx = invert_mel_spectrogram(mel_spectrogram, sample_rate)

    # Reconstruct the time-domain waveform
    reconstructed_audio = reconstruct_waveform(magnitude_spectrogram_approx)

    # Save the reconstructed audio
    generated_audio_path = save_audio(reconstructed_audio, sample_rate, filename)

    # Save the Mel spectrogram plot
    save_mel_spectrogram_plot(
        mel_spectrogram,
        sample_rate,
        filename,
    )

    # Save waveforms
    save_waveforms(original_audio_data, reconstructed_audio, sample_rate, filename)

    # Plot F0 contour comparison
    save_f0_contour(filename, original_audio_data, reconstructed_audio, sample_rate)

    # Perform objective evaluation (PESQ)
    perform_pesq_evaluation(original_audio_path, generated_audio_path)


def main() -> None:
    """
    Process all WAV files in the 'tts/audio_tts' directory by applying the tts function to each file.

    This function searches the 'tts/audio_tts' folder for all files with a '.wav' extension and
    processes each file using the previously defined `tts` function.

    Raises:
        FileNotFoundError: If the 'audio_tts' directory does not exist.
    """
    audio_directory = Path("tts/audio_tts")

    if not audio_directory.exists() or not audio_directory.is_dir():
        raise FileNotFoundError(
            f"The directory '{audio_directory}' does not exist or is not a directory."
        )

    wav_files = list(audio_directory.glob("*.wav"))

    for wav_file in wav_files:
        print(f"Processing file: {wav_file}")
        tts_pipeline(str(wav_file))


if __name__ == "__main__":
    main()
