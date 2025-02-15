import wx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from scipy.io.wavfile import write as write_wav
import tensorflow as tf

# Load the saved generator model
generator = tf.keras.models.load_model('generator_final.h5')
latent_dim = 10  # Latent dimension used during training

class SignalGeneratorApp(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title="Signal Generator", size=(800, 600))
        self.panel = wx.Panel(self)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.panel, -1, self.figure)

        # Buttons
        self.generate_button = wx.Button(self.panel, label="Generate Signal", pos=(20, 500))
        self.save_button = wx.Button(self.panel, label="Save as Audio", pos=(200, 500))

        # Bind buttons to events
        self.generate_button.Bind(wx.EVT_BUTTON, self.on_generate)
        self.save_button.Bind(wx.EVT_BUTTON, self.on_save)

        # Initialize signal storage
        self.generated_signal = None

        # Layout
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 10)
        self.vbox.Add(self.generate_button, 0, wx.ALL | wx.CENTER, 10)
        self.vbox.Add(self.save_button, 0, wx.ALL | wx.CENTER, 10)
        self.panel.SetSizer(self.vbox)

    def on_generate(self, event):
        """Generate a new signal using the generator model."""
        # Generate random noise as input to the generator
        noise = np.random.normal(0, 1, (1000, latent_dim)).astype(np.float32)  # Generate 1000 samples
        self.generated_signal = generator(noise, training=False).numpy().flatten()

        # Plot the generated signal
        self.ax.clear()
        self.ax.plot(self.generated_signal, label="Generated Signal")
        self.ax.set_title("Generated Signal")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.canvas.draw()

    def on_save(self, event):
        """Save the generated signal as an audio file."""
        if self.generated_signal is None:
            wx.MessageBox("No signal generated yet!", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Normalize the signal to the range [-1, 1] for audio
        normalized_signal = self.generated_signal / np.max(np.abs(self.generated_signal))

        # Save as a WAV file
        file_dialog = wx.FileDialog(self, "Save Audio File", wildcard="WAV files (*.wav)|*.wav",
                                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        file_path = file_dialog.GetPath()
        sample_rate = 44100  # Standard sample rate for audio
        write_wav(file_path, sample_rate, normalized_signal)
        wx.MessageBox(f"Signal saved to {file_path}", "Success", wx.OK | wx.ICON_INFORMATION)

if __name__ == "__main__":
    app = wx.App(False)
    frame = SignalGeneratorApp()
    frame.Show()
    app.MainLoop()