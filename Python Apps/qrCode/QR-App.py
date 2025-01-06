import wx
import cv2
import qrcode
from PIL import Image
import numpy as np


class QRApp(wx.Frame):
    def __init__(self, parent, title):
        super(QRApp, self).__init__(parent, title=title, size=(800, 600))

        # Main Panel
        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)

        # Tabs
        self.generator_tab = QRGeneratorPanel(notebook)
        self.decoder_tab = QRDecoderPanel(notebook)

        notebook.AddPage(self.generator_tab, "QR Generator")
        notebook.AddPage(self.decoder_tab, "QR Decoder")

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(notebook, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)
        panel.SetSizer(vbox)

        self.Centre()
        self.Show()


class QRGeneratorPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Input
        self.text_input = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        vbox.Add(wx.StaticText(self, label="Enter text or URL:"), flag=wx.LEFT, border=10)
        vbox.Add(self.text_input, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Color Options
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.fill_color = wx.ColourPickerCtrl(self)
        self.back_color = wx.ColourPickerCtrl(self)
        hbox.Add(wx.StaticText(self, label="Foreground Color:"), flag=wx.RIGHT, border=5)
        hbox.Add(self.fill_color, flag=wx.RIGHT, border=20)
        hbox.Add(wx.StaticText(self, label="Background Color:"), flag=wx.RIGHT, border=5)
        hbox.Add(self.back_color)
        vbox.Add(hbox, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Generate Button
        self.generate_button = wx.Button(self, label="Generate QR Code")
        self.generate_button.Bind(wx.EVT_BUTTON, self.on_generate)
        vbox.Add(self.generate_button, flag=wx.EXPAND | wx.ALL, border=10)

        # Output Image
        self.image_display = wx.StaticBitmap(self)
        vbox.Add(self.image_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        self.SetSizer(vbox)

    def on_generate(self, event):
        text = self.text_input.GetValue()
        if not text:
            wx.MessageBox("Please enter text or a URL.", "Error", wx.ICON_ERROR)
            return

        # Get colors
        fill_color = self.fill_color.GetColour().GetAsString(wx.C2S_HTML_SYNTAX)
        back_color = self.back_color.GetColour().GetAsString(wx.C2S_HTML_SYNTAX)

        # Generate QR Code
        qr_image = self.generate_qr(text, fill_color, back_color)
        qr_image.save("output_qr.png")

        # Display QR Code
        bitmap = wx.Bitmap("output_qr.png", wx.BITMAP_TYPE_PNG)
        self.image_display.SetBitmap(bitmap)
        self.Refresh()

    def generate_qr(self, data, fill_color="black", back_color="white"):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        # Make image with specified colors
        img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")
        return img


class QRDecoderPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)

        # File Selection
        self.file_path_text = wx.TextCtrl(self, style=wx.TE_READONLY)
        vbox.Add(wx.StaticText(self, label="Select QR Code Image:"), flag=wx.LEFT, border=10)
        vbox.Add(self.file_path_text, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        self.browse_button = wx.Button(self, label="Browse Image")
        self.browse_button.Bind(wx.EVT_BUTTON, self.on_browse)
        vbox.Add(self.browse_button, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Decode Button
        self.decode_button = wx.Button(self, label="Decode QR Code")
        self.decode_button.Bind(wx.EVT_BUTTON, self.on_decode)
        vbox.Add(self.decode_button, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Decoded Data
        self.output_text = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        vbox.Add(wx.StaticText(self, label="Decoded QR Code Data:"), flag=wx.LEFT | wx.TOP, border=10)
        vbox.Add(self.output_text, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        # Display Decoded Image
        self.image_display = wx.StaticBitmap(self)
        vbox.Add(self.image_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        self.SetSizer(vbox)

    def on_browse(self, event):
        with wx.FileDialog(self, "Choose a QR code image", wildcard="Image files (*.png;*.jpg)|*.png;*.jpg",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                self.file_path_text.SetValue(dialog.GetPath())

    def on_decode(self, event):
        image_path = self.file_path_text.GetValue()
        if not image_path:
            wx.MessageBox("Please select an image file.", "Error", wx.ICON_ERROR)
            return

        # Decode QR Code
        data = self.decode_qr(image_path)
        self.output_text.SetValue(data)

        # Display Image
        self.display_image(image_path)

    def decode_qr(self, image_path):
        detector = cv2.QRCodeDetector()
        img = cv2.imread(image_path)
        data, _, _ = detector.detectAndDecode(img)
        return data if data else "No QR code detected."

    def display_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        wx_img = wx.Bitmap.FromBuffer(w, h, img)
        self.image_display.SetBitmap(wx_img)
        self.Refresh()


if __name__ == "__main__":
    app = wx.App(False)
    frame = QRApp(None, title="QR Code Generator & Decoder")
    app.MainLoop()
