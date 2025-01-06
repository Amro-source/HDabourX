# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:23:22 2024

@author: M5
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        
        # Create a listbox with multiple options
        self.listbox = wx.ListBox(panel, choices=["Item 1", "Item 2", "Item 3", "Item 4"], pos=(20, 30), size=(200, 100))
        
        # Create a button to get selected items
        button = wx.Button(panel, label="Get Selected Item", pos=(20, 140))
        button.Bind(wx.EVT_BUTTON, self.OnListBox)

        self.SetTitle("ListBox Example")
        self.SetSize(300, 200)
        self.Centre()

    def OnListBox(self, event):
        selection = self.listbox.GetSelection()
        if selection != wx.NOT_FOUND:
            selected_item = self.listbox.GetString(selection)
            wx.MessageBox(f"Selected: {selected_item}", "Info", wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox("No item selected", "Info", wx.OK | wx.ICON_WARNING)

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="wxPython ListBox Example")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
