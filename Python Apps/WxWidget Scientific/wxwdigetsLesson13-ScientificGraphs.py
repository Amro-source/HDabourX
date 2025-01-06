# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:50:44 2024

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

        # TextBox for entering the equation
        self.text_box = wx.TextCtrl(panel, size=(400, 30), pos=(20, 20))

        # Button to plot the equation
        self.plot_button = wx.Button(panel, label="Plot Equation", pos=(20, 60))
        self.plot_button.Bind(wx.EVT_BUTTON, self.OnPlot)

        # Placeholder for the plot
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.canvas.SetSize((500, 350))
        self.canvas.SetPosition((20, 100))

        self.SetTitle("Algebraic Equation Plotter")
        self.SetSize(600, 500)
        self.Centre()

    def OnPlot(self, event):
        # Get the equation from the TextCtrl
        equation = self.text_box.GetValue()
        if not equation:
            wx.MessageBox("Please enter an equation!", "Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            # Generate x values (we can choose a range)
            x = np.linspace(-10, 10, 400)
            
            # Evaluate the equation: 
            # We use lambda to evaluate the equation string as a Python function.
            y = eval(equation)

            # Create a subplot
            ax = self.figure.add_subplot(111)

            # Clear the previous plot
            ax.clear()

            # Plot the equation
            ax.plot(x, y, label=f'Plot of: {equation}')
            ax.set_title(f'Plot of: {equation}')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            ax.legend()

            # Refresh the canvas to show the plot
            self.canvas.draw()

        except Exception as e:
            wx.MessageBox(f"Error in the equation: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Algebraic Equation Plotter")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
