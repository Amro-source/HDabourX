# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:48:15 2024

@author: M5
"""

import wx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)

        # Create a button that triggers plotting
        self.plot_button = wx.Button(panel, label="Plot Graph", pos=(10, 10))
        self.plot_button.Bind(wx.EVT_BUTTON, self.OnPlot)

        # Placeholder for the plot
        self.figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.canvas.SetSize((500, 300))
        self.canvas.SetPosition((10, 50))

        self.SetTitle("wxPython with Matplotlib")
        self.SetSize(600, 400)
        self.Centre()

    def OnPlot(self, event):
        # Create a subplot
        ax = self.figure.add_subplot(111)
        
        # Generate some sample data (sine wave)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # Plot the data
        ax.plot(x, y, label="Sine Wave")
        ax.set_title("Sine Wave Example")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.legend()

        # Refresh the canvas to show the plot
        self.canvas.draw()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython & Matplotlib Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
