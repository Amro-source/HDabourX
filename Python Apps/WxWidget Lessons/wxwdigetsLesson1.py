# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:46:03 2024

@author: M5
"""
import wx

# Define the main frame class
class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        self.SetTitle("Simple wxWidgets Window")
        self.SetSize(300, 200)
        self.Centre()

# Define the main app class
class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="Hello wxPython", size=(300, 200))
        self.frame.Show(True)
        return True

# Run the app
if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()

