# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:05:29 2024

@author: Meshmesh
"""

import wx
import cv2
import numpy as np


class QRDecoderApp(wx.Frame):
    def __init__(self, parent, title):
        super(QRDecoderApp, self).__init__(parent, title=title, size=(600, 500))

        # Panel
        panel = wx.Panel(self)

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Title Text
        self.title = wx.StaticText(panel, label="QR Code Decoder", style=wx.ALIGN_CENTER)
        font = self.title.GetFont()
        font.PointSize += 10
        font = font.Bold()
        self.title.SetFont(font)
        vbox.Add(self.title, flag=wx.EXPAND | wx.ALL, border=10)

        # Image Path Text
        self.file_path_text = wx.TextCtrl(panel, style=wx.TE_READONLY)
        vbox.Add(self.file_path_text, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Browse Button
        self.browse_button = wx.Button(panel, label="Browse Image")
        self.browse_button.Bind(wx.EVT_BUTTON, self.on_browse)
        vbox.Add(self.browse_button, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Decode Button
        self.decode_button = wx.Button(panel, label="Decode QR Code")
        self.decode_button.Bind(wx.EVT_BUTTON, self.on_decode)
        vbox.Add(self.decode_button, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # QR Code Data Output
        self.output_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        vbox.Add(self.output_text, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        # Image Display Area
        self.image_display = wx.StaticBitmap(panel)
        vbox.Add(self.image_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Set panel sizer
        panel.SetSizer(vbox)

        # Show the application
        self.Show()

    def on_browse(self, event):
        # Open file dialog to select an image
        file_dialog = wx.FileDialog(self, "Open Image", wildcard="Image files (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg",
                                    style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if file_dialog.ShowModal() == wx.ID_CANCEL:
            return  # Cancelled by user

        # Get the selected file path
        path = file_dialog.GetPath()
        self.file_path_text.SetValue(path)

    def on_decode(self, event):
        # Get the file path
        image_path = self.file_path_text.GetValue()
        if not image_path:
            wx.MessageBox("Please select an image file first!", "Error", wx.ICON_ERROR)
            return

        # Decode the QR code using OpenCV
        try:
            detector = cv2.QRCodeDetector()
            image = cv2.imread(image_path)

            if image is None:
                wx.MessageBox("Failed to load image. Please select a valid image file.", "Error", wx.ICON_ERROR)
                return

            # Detect and decode the QR code
            data, vertices, _ = detector.detectAndDecode(image)

            if data:
                self.output_text.SetValue(f"QR Code Data:\n{data}")

                # Draw the bounding box if vertices are detected
                if vertices is not None:
                    vertices = vertices[0]  # Extract the vertices array
                    for i in range(len(vertices)):
                        pt1 = tuple(map(int, vertices[i]))
                        pt2 = tuple(map(int, vertices[(i + 1) % len(vertices)]))
                        cv2.line(image, pt1, pt2, (0, 255, 0), 3)

            else:
                self.output_text.SetValue("No QR code detected in the selected image.")

            # Convert the image to wx.Bitmap for display
            self.display_image(image)

        except Exception as e:
            wx.MessageBox(f"An error occurred: {str(e)}", "Error", wx.ICON_ERROR)

    def display_image(self, image):
        # Convert OpenCV image (BGR) to wx.Bitmap
        height, width, channels = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image_wx = wx.Bitmap.FromBuffer(width, height, image_rgb)

        # Display the image in the StaticBitmap widget
        self.image_display.SetBitmap(image_wx)
        self.Refresh()  # Refresh the GUI


if __name__ == "__main__":
    app = wx.App(False)
    frame = QRDecoderApp(None, title="QR Code Decoder with GUI")
    app.MainLoop()
