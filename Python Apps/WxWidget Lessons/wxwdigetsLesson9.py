# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:20:23 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)

        # Create radio buttons
        self.radio1 = wx.RadioButton(panel, label="Option 1", pos=(20, 30), style=wx.RB_GROUP)
        self.radio2 = wx.RadioButton(panel, label="Option 2", pos=(20, 60))
        self.radio3 = wx.RadioButton(panel, label="Option 3", pos=(20, 90))

        # Create a button to get selected radio button
        button = wx.Button(panel, label="Get Selected Option", pos=(20, 130))
        button.Bind(wx.EVT_BUTTON, self.OnRadio)

        self.SetTitle("Radio Button Example")
        self.SetSize(300, 200)
        self.Centre()

    def OnRadio(self, event):
        selected = None
        if self.radio1.GetValue():
            selected = "Option 1"
        elif self.radio2.GetValue():
            selected = "Option 2"
        elif self.radio3.GetValue():
            selected = "Option 3"
        
        wx.MessageBox(f"Selected: {selected}", "Info", wx.OK | wx.ICON_INFORMATION)

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Radio Button Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
