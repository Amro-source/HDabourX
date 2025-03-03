import os
import wx
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ✅ 1. Define the Generator Model (Must Match Training Code)
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  

        x = F.relu(self.ct1(x))  
        x = F.relu(self.ct2(x))  

        x = torch.tanh(self.conv(x))  
        return x

# ✅ 2. Function to Load Trained Generator Model
def load_generator(model_path="checkpoints/generator.pth", latent_dim=100):
    generator = Generator(latent_dim)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    generator.eval()
    return generator

# ✅ 3. Function to Generate and Save Images
def generate_and_save_images(generator, num_images=10, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    latent_dim = 100

    for i in range(num_images):
        z = torch.randn(1, latent_dim)
        with torch.no_grad():
            img_tensor = generator(z).cpu()

        print(f"Generated image shape: {img_tensor.shape}")  # Debugging

        img_tensor = img_tensor.squeeze().numpy()  # Correctly reshape

        img = (img_tensor * 255).astype(np.uint8)  # Normalize to grayscale
        img = Image.fromarray(img, "L")  # Convert to PIL Image

        img_path = os.path.join(output_dir, f"image_{i+1}.png")
        img.save(img_path)
        print(f"Saved {img_path}")

    return output_dir



# ✅ 4. wxPython GUI for GAN Image Generation
class GANApp(wx.Frame):
    def __init__(self, *args, **kw):
        super(GANApp, self).__init__(*args, **kw)
        
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.title = wx.StaticText(panel, label="GAN Image Generator", style=wx.ALIGN_CENTER)
        font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.title.SetFont(font)

        self.generate_btn = wx.Button(panel, label="Generate Images")
        self.generate_btn.Bind(wx.EVT_BUTTON, self.on_generate)

        self.image_list = wx.ListBox(panel, size=(300, 200))

        vbox.Add(self.title, flag=wx.ALL | wx.CENTER, border=10)
        vbox.Add(self.generate_btn, flag=wx.ALL | wx.CENTER, border=10)
        vbox.Add(self.image_list, flag=wx.ALL | wx.CENTER, border=10)

        panel.SetSizer(vbox)
        self.generator = load_generator()  

    def on_generate(self, event):
        output_dir = generate_and_save_images(self.generator)
        images = os.listdir(output_dir)
        self.image_list.Set(images)  
        wx.MessageBox("Images Generated Successfully!", "Success", wx.OK | wx.ICON_INFORMATION)

if __name__ == "__main__":
    app = wx.App(False)
    frame = GANApp(None, title="GAN Image Generator", size=(400, 400))
    frame.Show()
    app.MainLoop()
