# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:08:28 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        # Create a menu bar
        menubar = wx.MenuBar()
        
        # Create a file menu
        fileMenu = wx.Menu()
        openItem = fileMenu.Append(wx.ID_OPEN, 'Open', 'Open a file')
        exitItem = fileMenu.Append(wx.ID_EXIT, 'Exit', 'Exit application')
        
        # Bind the events to the menu items
        self.Bind(wx.EVT_MENU, self.OnOpen, openItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        
        # Add the menu bar to the frame
        menubar.Append(fileMenu, 'File')
        self.SetMenuBar(menubar)

        self.SetTitle("Menu Example")
        self.SetSize(300, 200)
        self.Centre()

    def OnOpen(self, event):
        wx.MessageBox('Open menu item clicked', 'Info', wx.OK | wx.ICON_INFORMATION)

    def OnExit(self, event):
        self.Close()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython Menu Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
