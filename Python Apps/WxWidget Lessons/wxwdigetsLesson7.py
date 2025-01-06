# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:18:09 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        
        # Create a slider (range from 0 to 100)
        self.slider = wx.Slider(panel, value=50, minValue=0, maxValue=100, pos=(50, 30), size=(250, -1))
        self.slider.Bind(wx.EVT_SLIDER, self.OnSlider)

        # Label to show the slider value
        self.label = wx.StaticText(panel, label="Value: 50", pos=(50, 80))

        self.SetTitle("Slider Example")
        self.SetSize(350, 200)
        self.Centre()

    def OnSlider(self, event):
        value = self.slider.GetValue()  # Get current value from slider
        self.label.SetLabel(f"Value: {value}")  # Update label with the slider value

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Slider Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
