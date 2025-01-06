# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:10:25 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        
        # Create a button to trigger the file dialog
        open_button = wx.Button(panel, label="Open File", pos=(150, 30))
        open_button.Bind(wx.EVT_BUTTON, self.OnOpenFile)

        self.SetTitle("File Dialog Example")
        self.SetSize(300, 200)
        self.Centre()

    def OnOpenFile(self, event):
        with wx.FileDialog(self, "Open file", wildcard="Text files (*.txt)|*.txt", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            path = fileDialog.GetPath()
            wx.MessageBox(f'Selected file: {path}', 'Info', wx.OK | wx.ICON_INFORMATION)

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython File Dialog Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
