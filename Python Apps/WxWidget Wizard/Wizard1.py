# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:24:52 2024

@author: M5
"""

import wx

class WizardPage(wx.Panel):
    def __init__(self, parent, title):
        super(WizardPage, self).__init__(parent)
        self.title = title
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.label = wx.StaticText(self, label=title)
        self.sizer.Add(self.label, 0, wx.ALL | wx.CENTER, 5)
        self.SetSizer(self.sizer)

class Wizard(wx.Frame):
    def __init__(self):
        super(Wizard, self).__init__(None, title="Simple Wizard", size=(400, 300))
        
        self.pages = [
            WizardPage(self, "Step 1: Welcome to the Wizard!"),
            WizardPage(self, "Step 2: Please provide your information."),
            WizardPage(self, "Step 3: Finish!")
        ]
        
        self.current_page = 0
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.pages[self.current_page], 1, wx.EXPAND | wx.ALL, 5)
        
        self.next_button = wx.Button(self, label="Next")
        self.prev_button = wx.Button(self, label="Previous")
        self.finish_button = wx.Button(self, label="Finish")
        
        self.next_button.Bind(wx.EVT_BUTTON, self.on_next)
        self.prev_button.Bind(wx.EVT_BUTTON, self.on_prev)
        self.finish_button.Bind(wx.EVT_BUTTON, self.on_finish)
        
        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.button_sizer.Add(self.prev_button, 0, wx.ALL, 5)
        self.button_sizer.Add(self.next_button, 0, wx.ALL, 5)
        self.button_sizer.Add(self.finish_button, 0, wx.ALL, 5)
        
        self.sizer.Add(self.button_sizer, 0, wx.ALIGN_CENTER)
        self.SetSizer(self.sizer)
        
        self.update_buttons()
        
    def on_next(self, event):
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
            self.update_wizard()
    
    def on_prev(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_wizard()
    
    def on_finish(self, event):
        wx.MessageBox("Wizard Finished!", "Info", wx.OK | wx.ICON_INFORMATION)
        self.Close()
    
    def update_wizard(self):
        self.sizer.Hide(self.pages[self.current_page - 1])
        self.sizer.Show(self.pages[self.current_page])
        self.Layout()
        self.update_buttons()
    
    def update_buttons(self):
        self.prev_button.Enable(self.current_page > 0)
        self.next_button.Enable(self.current_page < len(self.pages) - 1)
        self.finish_button.Enable(self.current_page == len(self.pages) - 1)

if __name__ == "__main__":
    app = wx.App(False)
    wizard = Wizard()
    wizard.Show()
    app.MainLoop()