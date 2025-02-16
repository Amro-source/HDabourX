import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Generate real data (tones)
def generate_real_tones(n_samples, signal_length=44100):
    X = np.zeros((n_samples, signal_length))
    for i in range(n_samples):
        frequency = np.random.uniform(100, 2000)  # Wider frequency range
        amplitude = np.random.uniform(0.5, 1.0)   # Random amplitude
        phase = np.random.uniform(0, 2 * np.pi)   # Random phase
        t = np.linspace(0, 1, signal_length, endpoint=False)
        X[i] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return X

# Plot a single tone in time and frequency domains
def plot_tone(tone, sample_rate=44100):
    plt.figure(figsize=(12, 6))

    # Time-domain plot
    plt.subplot(2, 1, 1)
    t = np.linspace(0, 1, len(tone), endpoint=False)
    plt.plot(t, tone, color='blue', linewidth=0.5)
    plt.title("Tone Waveform (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Frequency-domain plot
    plt.subplot(2, 1, 2)
    fft_signal = fft(tone)
    frequencies = fftfreq(len(tone), 1 / sample_rate)
    plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft_signal)[:len(fft_signal) // 2], color='blue')
    plt.title("Frequency Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate a single tone for visualization
    signal_length = 44100
    tone = generate_real_tones(1, signal_length=signal_length)[0]

    # Plot the tone
    plot_tone(tone, sample_rate=44100)