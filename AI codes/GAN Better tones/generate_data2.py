import os
import numpy as np
from scipy.io.wavfile import write as write_wav
import wx

# Function to generate real tones (sine waves)
def generate_real_tones(n_samples, min_freq=100, max_freq=2000, signal_length=44100):
    X = np.zeros((n_samples, signal_length))
    for i in range(n_samples):
        frequency = np.random.uniform(min_freq, max_freq)  # Random frequency within range
        amplitude = np.random.uniform(0.5, 1.0)           # Random amplitude
        phase = np.random.uniform(0, 2 * np.pi)           # Random phase
        t = np.linspace(0, 1, signal_length, endpoint=False)
        X[i] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return X

# Function to save tones as audio files
def save_tones_as_audio(tones, output_dir="generated_tones", sample_rate=44100):
    os.makedirs(output_dir, exist_ok=True)
    for i, tone in enumerate(tones):
        normalized_signal = (tone / np.max(np.abs(tone))) * 32767
        normalized_signal = np.clip(normalized_signal, -32767, 32767).astype(np.int16)
        output_path = os.path.join(output_dir, f"tone_{i + 1}.wav")
        write_wav(output_path, sample_rate, normalized_signal)
        print(f"Tone saved to {output_path}.")

# GUI for generating and saving tones
class ToneGeneratorApp(wx.App):
    def OnInit(self):
        # Initialize the main frame
        frame = wx.Frame(None, title="Tone Generator", size=(400, 300))
        panel = wx.Panel(frame)

        # Create layout elements
        self.num_samples_label = wx.StaticText(panel, label="Number of Samples:")
        self.num_samples_input = wx.TextCtrl(panel, value="10")  # Default value: 10 samples

        self.min_freq_label = wx.StaticText(panel, label="Minimum Frequency (Hz):")
        self.min_freq_input = wx.TextCtrl(panel, value="100")  # Default value: 100 Hz

        self.max_freq_label = wx.StaticText(panel, label="Maximum Frequency (Hz):")
        self.max_freq_input = wx.TextCtrl(panel, value="2000")  # Default value: 2000 Hz

        self.generate_button = wx.Button(panel, label="Generate and Save Tones")

        # Bind events
        self.generate_button.Bind(wx.EVT_BUTTON, self.on_generate_and_save)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.num_samples_label, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.num_samples_input, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.min_freq_label, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.min_freq_input, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.max_freq_label, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.max_freq_input, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.generate_button, 0, wx.ALL | wx.CENTER, 10)

        panel.SetSizer(sizer)
        frame.Show()
        return True

    def on_generate_and_save(self, event):
        """Event handler for generating and saving tones."""
        try:
            num_samples = int(self.num_samples_input.GetValue())
            if num_samples <= 0:
                raise ValueError("Number of samples must be greater than 0.")

            min_freq = int(self.min_freq_input.GetValue())
            max_freq = int(self.max_freq_input.GetValue())
            if min_freq >= max_freq or min_freq <= 0:
                raise ValueError("Invalid frequency range. Ensure min_freq < max_freq and min_freq > 0.")
        except ValueError as e:
            wx.MessageBox(f"Invalid input: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Generate tones
        signal_length = 44100
        tones = generate_real_tones(num_samples, min_freq=min_freq, max_freq=max_freq, signal_length=signal_length)

        # Save tones as audio files
        output_dir = "generated_tones"
        save_tones_as_audio(tones, output_dir=output_dir)

        # Show success message
        wx.MessageBox(
            f"{num_samples} tones (frequency range: {min_freq}-{max_freq} Hz) generated and saved to {output_dir}.",
            "Success",
            wx.OK | wx.ICON_INFORMATION
        )

if __name__ == "__main__":
    app = ToneGeneratorApp()
    app.MainLoop()