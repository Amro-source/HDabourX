# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:04:50 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        
        # Create a label and text input field
        self.label = wx.StaticText(panel, label="Enter your name:", pos=(20, 30))
        self.text_ctrl = wx.TextCtrl(panel, pos=(150, 30), size=(200, 30))
        
        # Create a button
        button = wx.Button(panel, label="Submit", pos=(150, 80))
        button.Bind(wx.EVT_BUTTON, self.OnSubmit)

        # Result text
        self.result_text = wx.StaticText(panel, label="", pos=(150, 120))

        self.SetTitle("Text Input Example")
        self.SetSize(400, 200)
        self.Centre()

    def OnSubmit(self, event):
        name = self.text_ctrl.GetValue()  # Get the value from the text input field
        self.result_text.SetLabel(f"Hello, {name}!")  # Update the result label

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Text Input")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
