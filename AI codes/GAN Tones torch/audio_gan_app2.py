import os
import numpy as np
import torch
import torch.nn as nn  # Import torch.nn for model definition
from scipy.io.wavfile import write as write_wav
import wx
import wx.media

# Load the pre-trained generator model
class AudioGANApp(wx.App):
    def __init__(self, generator_path, output_dir="output"):
        self.generator_path = generator_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        super().__init__()

    def OnInit(self):
        # Initialize the main frame
        frame = wx.Frame(None, title="Audio GAN App", size=(400, 300))
        panel = wx.Panel(frame)

        # Load the generator model
        try:
            self.device = torch.device("cpu")
            self.latent_dim = 10
            self.signal_length = 44100
            self.generator = self.load_generator()
        except Exception as e:
            wx.MessageBox(f"Failed to load the generator: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return False

        # Create buttons
        generate_button = wx.Button(panel, label="Generate Tone")
        save_button = wx.Button(panel, label="Save Tone")
        play_button = wx.Button(panel, label="Play Tone")

        # Bind events
        generate_button.Bind(wx.EVT_BUTTON, self.on_generate_tone)
        save_button.Bind(wx.EVT_BUTTON, self.on_save_tone)
        play_button.Bind(wx.EVT_BUTTON, self.on_play_tone)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(generate_button, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(save_button, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(play_button, 0, wx.ALL | wx.EXPAND, 5)

        panel.SetSizer(sizer)
        frame.Show()
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