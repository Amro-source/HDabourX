# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:50:47 2024

@author: M5
"""

import wx
import cv2
import easyocr
import numpy as np
from PIL import Image

class VideoOCRApp(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.total_frames = 0
        self.mark_text = False  # Flag for marking text regions
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

    def init_ui(self):
        panel = wx.Panel(self)

        # File picker
        self.file_btn = wx.Button(panel, label="Load Video")
        self.file_btn.Bind(wx.EVT_BUTTON, self.load_video)

        # Slider
        self.slider = wx.Slider(panel, value=0, minValue=0, maxValue=100, style=wx.SL_HORIZONTAL)
        self.slider.Bind(wx.EVT_SLIDER, self.on_slide)

        # Video display
        self.video_display = wx.StaticBitmap(panel, size=(640, 360))

        # Checkbox for marking text
        self.mark_checkbox = wx.CheckBox(panel, label="Mark Text Regions")
        self.mark_checkbox.Bind(wx.EVT_CHECKBOX, self.toggle_mark_text)

        # OCR button
        self.ocr_btn = wx.Button(panel, label="Perform OCR")
        self.ocr_btn.Bind(wx.EVT_BUTTON, self.perform_ocr)

        # Output text
        self.text_output = wx.TextCtrl(panel, style=wx.TE_MULTILINE)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.file_btn, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.slider, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.video_display, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.mark_checkbox, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.ocr_btn, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.text_output, 1, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(sizer)
        self.SetSize((800, 600))
        self.SetTitle("Video OCR App")
        self.Centre()

    def toggle_mark_text(self, event):
        """Toggle the text marking feature."""
        self.mark_text = self.mark_checkbox.IsChecked()

    def load_video(self, event):
        with wx.FileDialog(self, "Open Video File", wildcard="Video files (*.mp4;*.avi)|*.mp4;*.avi",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            self.video_path = dialog.GetPath()
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                wx.MessageBox("Failed to open video file.", "Error", wx.ICON_ERROR)
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.SetMax(self.total_frames - 1)
            self.show_frame(0)

    def show_frame(self, frame_no):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        if not ret:
            wx.MessageBox("Unable to read frame from video. Check the file format or codec.", "Error", wx.ICON_ERROR)
            return

        # Store the current frame for OCR
        self.current_frame = frame

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to fit widget size
        widget_width, widget_height = self.video_display.GetSize()
        frame_rgb = cv2.resize(frame_rgb, (widget_width, widget_height), interpolation=cv2.INTER_AREA)

        # Convert to wx.Image and update StaticBitmap
        h, w = frame_rgb.shape[:2]
        img = wx.Image(w, h, frame_rgb.tobytes())
        self.video_display.SetBitmap(wx.Bitmap(img))
        self.video_display.Refresh()

    def on_slide(self, event):
        frame_no = self.slider.GetValue()
        self.show_frame(frame_no)

    def perform_ocr(self, event):
        if self.current_frame is None:
            wx.MessageBox("No frame selected for OCR.", "Error", wx.ICON_ERROR)
            return

        #
        
        # Convert the current frame to grayscale for better OCR results
        gray_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # Perform OCR using EasyOCR
        results = self.reader.readtext(gray_frame)

        # Initialize a variable to store formatted text
        formatted_text = ""
        marked_frame = self.current_frame.copy()

        for (bbox, text, prob) in results:
            if text.strip():
                # Retrieve position info
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x, y, w, h = int(top_left[0]), int(top_left[1]), int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])

                # Append to formatted output
                formatted_text += f'<div style="position:absolute; left:{x}px; top:{y}px; width:{w}px; height:{h}px;">{text}</div>\n'

                # Mark text regions with blue boxes if enabled
                if self.mark_text:
                    cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display marked frame
        if self.mark_text:
            frame_rgb = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
            widget_width, widget_height = self.video_display.GetSize()
            frame_rgb = cv2.resize(frame_rgb, (widget_width, widget_height), interpolation=cv2.INTER_AREA)
            h, w = frame_rgb.shape[:2]
            img = wx.Image(w, h, frame_rgb.tobytes())
            self.video_display.SetBitmap(wx.Bitmap(img))
            self.video_display.Refresh()

        # Save the output to an HTML file with improved formatting
        output_path = "output.html"
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    position: relative;
                    font-family: Arial, sans-serif;
                }}
                div {{
                    position: absolute;
                    color: black;
                    background-color: rgba(255, 255, 255, 0.8);
                    border: 1px solid #000;
                    padding: 2px;
                }}
            </style>
            <title>OCR Output</title>
        </head>
        <body>
            {formatted_text}
        </body>
        </html>
        """
        with open(output_path, "w") as f:
            f.write(html_content)

        # Update the UI with OCR result
        self.text_output.SetValue("OCR performed. Output saved as HTML.")
        wx.MessageBox(f"OCR complete. Saved to {output_path}", "Information", wx.ICON_INFORMATION)


if __name__ == "__main__":
    # Launch the app
    app = wx.App(False)
    frame = VideoOCRApp(None)
    frame.Show()
    app.MainLoop()