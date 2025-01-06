# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:49:00 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        self.SetTitle("Button Event Handling")
        self.SetSize(400, 300)

        # Create a button
        self.button = wx.Button(panel, label="Click Me", pos=(150, 120))

        # Bind the button click event to the event handler
        self.button.Bind(wx.EVT_BUTTON, self.OnClick)

        self.Centre()

    def OnClick(self, event):
        wx.MessageBox("Button Clicked!", "Info", wx.OK | wx.ICON_INFORMATION)

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Button Event")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
