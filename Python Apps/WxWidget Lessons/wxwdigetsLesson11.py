# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:25:33 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        
        # Create a button to open the custom dialog
        button = wx.Button(panel, label="Open Dialog", pos=(100, 50))
        button.Bind(wx.EVT_BUTTON, self.OnDialog)

        self.SetTitle("Custom Dialog Example")
        self.SetSize(300, 200)
        self.Centre()

    def OnDialog(self, event):
        dialog = wx.TextEntryDialog(None, "Enter something:", "Custom Dialog", "")
        
        if dialog.ShowModal() == wx.ID_OK:
            user_input = dialog.GetValue()
            wx.MessageBox(f'You entered: {user_input}', 'Info', wx.OK | wx.ICON_INFORMATION)

        dialog.Destroy()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Custom Dialog Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
