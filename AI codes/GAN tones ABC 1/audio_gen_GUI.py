import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import wx
import torch.nn as nn  # Import torch.nn

# Hyperparameters
latent_dim = 10
signal_length = 44100  # Assuming all audio files are 1 second long at 44.1 kHz

# Build the Generator (Match the architecture used during training)
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Layer 1: Input -> 256 neurons
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),        # Layer 2: 256 -> 512 neurons
            nn.LeakyReLU(0.2),
            nn.Linear(512, signal_length),  # Layer 3: 512 -> signal_length neurons
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Load Models
def load_models(generator_path="models/generator_final.pth", discriminator_path="models/discriminator_final.pth"):
    generator = Generator(latent_dim)
    discriminator = None  # Not used in this GUI, but you can load it if needed
    generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
    generator.eval()
    return generator, discriminator

# Generate Audio Signal
def generate_signal(generator, latent_dim=latent_dim):
    noise = torch.randn(1, latent_dim)
    with torch.no_grad():
        signal = generator(noise).numpy().flatten()
    return signal

# Save Generated Signals
def save_signals(signals, output_dir="output", num_samples=1):
    os.makedirs(output_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        file_path = os.path.join(output_dir, f"generated_signal_{i + 1}.wav")
        scaled_signal = np.int16(signal / np.max(np.abs(signal)) * 32767)  # Scale to 16-bit PCM
        from scipy.io.wavfile import write
        write(file_path, 44100, scaled_signal)
    print(f"{num_samples} signals saved to {output_dir}")

# Main GUI Class
class GANApp(wx.App):
    def OnInit(self):
        self.frame = GANFrame(None, title="GAN Audio Generator")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True

# GUI Frame
class GANFrame(wx.Frame):
    def __init__(self, parent, title):
        super(GANFrame, self).__init__(parent, title=title, size=(800, 600))

        # Load models
        try:
            self.generator, _ = load_models()
        except Exception as e:
            wx.MessageBox(f"Error loading models: {e}", "Error")
            return

        # Create GUI elements
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Plot area
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.ax = self.figure.add_subplot(111)
        vbox.Add(self.canvas, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Controls
        hbox_controls = wx.BoxSizer(wx.HORIZONTAL)

        # Generate Button
        btn_generate = wx.Button(panel, label="Generate Signal")
        btn_generate.Bind(wx.EVT_BUTTON, self.on_generate)
        hbox_controls.Add(btn_generate, flag=wx.ALL, border=5)

        # Plot Button
        btn_plot = wx.Button(panel, label="Plot Signal")
        btn_plot.Bind(wx.EVT_BUTTON, self.on_plot)
        hbox_controls.Add(btn_plot, flag=wx.ALL, border=5)

        # Save Button
        self.txt_num_samples = wx.TextCtrl(panel, value="1", size=(50, -1))
        hbox_controls.Add(self.txt_num_samples, flag=wx.ALL, border=5)

        btn_save = wx.Button(panel, label="Save Signals")
        btn_save.Bind(wx.EVT_BUTTON, self.on_save)
        hbox_controls.Add(btn_save, flag=wx.ALL, border=5)

        vbox.Add(hbox_controls, flag=wx.CENTER | wx.ALL, border=10)

        panel.SetSizer(vbox)

        # Initialize variables
        self.generated_signal = None

    def on_generate(self, event):
        """Generate a new audio signal."""
        try:
            self.generated_signal = generate_signal(self.generator)
            wx.MessageBox("Signal generated!", "Info")
        except Exception as e:
            wx.MessageBox(f"Error generating signal: {e}", "Error")

    def on_plot(self, event):
        """Plot the generated signal."""
        if self.generated_signal is None:
            wx.MessageBox("No signal generated yet!", "Warning")
            return

        self.ax.clear()
        self.ax.plot(self.generated_signal, color='blue')
        self.ax.set_title("Generated Audio Signal")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def on_save(self, event):
        """Save generated signals to the output folder."""
        try:
            num_samples = int(self.txt_num_samples.GetValue())
            if num_samples <= 0:
                raise ValueError("Number of samples must be positive.")
        except ValueError:
            wx.MessageBox("Invalid number of samples!", "Error")
            return

        try:
            signals = [generate_signal(self.generator) for _ in range(num_samples)]
            save_signals(signals, output_dir="output", num_samples=num_samples)
            wx.MessageBox(f"{num_samples} signals saved to 'output' folder.", "Info")
        except Exception as e:
            wx.MessageBox(f"Error saving signals: {e}", "Error")

# Run the Application
if __name__ == "__main__":
    app = GANApp(False)
    app.MainLoop()