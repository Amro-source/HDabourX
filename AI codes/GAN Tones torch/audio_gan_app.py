import os
import numpy as np
import torch
import torch.nn as nn
from scipy.io.wavfile import write as write_wav
import wx
import wx.media
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

# Load the pre-trained generator model
class AudioGANApp(wx.App):
    def __init__(self, generator_path, output_dir="output"):
        self.generator_path = generator_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        super().__init__()

    def OnInit(self):
        # Initialize the main frame
        frame = wx.Frame(None, title="Audio GAN App", size=(800, 600))
        panel = wx.Panel(frame)

        # Create a vertical sizer for layout
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Add buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        generate_button = wx.Button(panel, label="Generate Tone")
        save_button = wx.Button(panel, label="Save Tone")
        play_button = wx.Button(panel, label="Play Tone")

        # Bind events
        generate_button.Bind(wx.EVT_BUTTON, self.on_generate_tone)
        save_button.Bind(wx.EVT_BUTTON, self.on_save_tone)
        play_button.Bind(wx.EVT_BUTTON, self.on_play_tone)

        # Add buttons to the button sizer
        button_sizer.Add(generate_button, 0, wx.ALL | wx.EXPAND, 5)
        button_sizer.Add(save_button, 0, wx.ALL | wx.EXPAND, 5)
        button_sizer.Add(play_button, 0, wx.ALL | wx.EXPAND, 5)

        # Add the button sizer to the main sizer
        main_sizer.Add(button_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Create a Matplotlib figure and canvas
        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Generated Tone Waveform")
        self.axes.set_xlabel("Sample Index")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True)

        # Add the canvas to the main sizer
        main_sizer.Add(self.canvas, 1, wx.ALL | wx.EXPAND, 5)

        # Set the main sizer for the panel
        panel.SetSizer(main_sizer)
        frame.Show()

        # Load the generator model
        try:
            self.device = torch.device("cpu")
            self.latent_dim = 10
            self.signal_length = 44100
            self.generator = self.load_generator()
        except Exception as e:
            wx.MessageBox(f"Failed to load the generator: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return False

        return True

    def load_generator(self):
        """Load the pre-trained generator model."""
        generator = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.signal_length),
            nn.Tanh()
        ).to(self.device)

        try:
            generator.load_state_dict(torch.load(self.generator_path, map_location=self.device))
            generator.eval()
            print("Generator loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load generator state dict: {e}")

        return generator

    def generate_tone(self):
        """Generate a tone using the generator."""
        noise = torch.randn(1, self.latent_dim, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            generated_signal = self.generator(noise).cpu().numpy().flatten()

        normalized_signal = (generated_signal / np.max(np.abs(generated_signal))) * 32767
        normalized_signal = np.clip(normalized_signal, -32767, 32767).astype(np.int16)
        self.generated_signal = normalized_signal
        print("Tone generated successfully.")

        # Update the plot
        self.update_plot()

    def save_tone(self):
        """Save the generated tone as an audio file."""
        if not hasattr(self, "generated_signal"):
            wx.MessageBox("Please generate a tone first.", "Error", wx.OK | wx.ICON_ERROR)
            return

        output_path = os.path.join(self.output_dir, "generated_tone.wav")
        try:
            write_wav(output_path, 44100, self.generated_signal)
            print(f"Tone saved to {output_path}.")
            wx.MessageBox(f"Tone saved to {output_path}.", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Failed to save tone: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def play_tone(self):
        """Play the generated tone."""
        if not hasattr(self, "generated_signal"):
            wx.MessageBox("Please generate a tone first.", "Error", wx.OK | wx.ICON_ERROR)
            return

        output_path = os.path.join(self.output_dir, "temp_tone.wav")
        try:
            write_wav(output_path, 44100, self.generated_signal)
        except Exception as e:
            wx.MessageBox(f"Failed to write temporary tone file: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Play the audio file
        if hasattr(self, "player") and self.player.IsPlaying():
            self.player.Stop()

        self.player = wx.media.MediaCtrl(self.TopWindow)
        if self.player.Load(output_path):
            self.player.Play()
        else:
            wx.MessageBox("Failed to load the audio file.", "Error", wx.OK | wx.ICON_ERROR)

    def update_plot(self):
        """Update the embedded plot with the generated tone."""
        if not hasattr(self, "generated_signal"):
            wx.MessageBox("Please generate a tone first.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Normalize the signal for plotting
        normalized_signal = self.generated_signal / 32767.0

        # Clear the previous plot
        self.axes.clear()

        # Plot the new waveform
        self.axes.plot(normalized_signal, color='blue', linewidth=0.5)
        self.axes.set_title("Generated Tone Waveform")
        self.axes.set_xlabel("Sample Index")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True)

        # Redraw the canvas
        self.canvas.draw()

    def on_generate_tone(self, event):
        """Event handler for generating a tone."""
        self.generate_tone()

    def on_save_tone(self, event):
        """Event handler for saving the tone."""
        self.save_tone()

    def on_play_tone(self, event):
        """Event handler for playing the tone."""
        self.play_tone()


if __name__ == "__main__":
    # Path to the pre-trained generator model
    generator_path = "models/generator_final.pth"

    # Check if the generator file exists
    if not os.path.exists(generator_path):
        print(f"Error: Generator file not found at {generator_path}.")
    else:
        # Start the app
        app = AudioGANApp(generator_path)
        app.MainLoop()