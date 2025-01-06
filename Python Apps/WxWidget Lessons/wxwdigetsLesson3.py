# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:01:33 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)

        # Create two buttons
        button1 = wx.Button(panel, label="Button 1")
        button2 = wx.Button(panel, label="Button 2")

        # Create a vertical box sizer
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(button1, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)
        vbox.Add(button2, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Set the panel layout to the vertical sizer
        panel.SetSizer(vbox)

        self.SetTitle("BoxSizer Layout")
        self.SetSize(300, 200)
        self.Centre()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Layout Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
