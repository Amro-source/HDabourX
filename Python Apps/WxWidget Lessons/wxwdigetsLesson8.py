# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:18:48 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        
        # Create checkboxes
        self.cb1 = wx.CheckBox(panel, label="Option 1", pos=(20, 30))
        self.cb2 = wx.CheckBox(panel, label="Option 2", pos=(20, 60))

        # Create a button to check the state of checkboxes
        button = wx.Button(panel, label="Check Options", pos=(20, 100))
        button.Bind(wx.EVT_BUTTON, self.OnCheck)

        self.SetTitle("Checkbox Example")
        self.SetSize(300, 200)
        self.Centre()

    def OnCheck(self, event):
        options = []
        if self.cb1.IsChecked():
            options.append("Option 1")
        if self.cb2.IsChecked():
            options.append("Option 2")
        
        if options:
            wx.MessageBox(f"Checked: {', '.join(options)}", "Info", wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox("No options selected", "Info", wx.OK | wx.ICON_INFORMATION)

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Checkbox Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
