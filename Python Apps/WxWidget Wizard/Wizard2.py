# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:19:32 2024

@author: M5
"""

import wx

class WizardPage(wx.Panel):
    def __init__(self, parent, title, content):
        super(WizardPage, self).__init__(parent)
        self.title = title
        self.content = content
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        self.label_title = wx.StaticText(self, label=self.title, style=wx.ALIGN_CENTER)
        self.label_title.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.sizer.Add(self.label_title, 0, wx.ALL | wx.CENTER, 10)

        # Content
        self.label_content = wx.StaticText(self, label=self.content)
        self.sizer.Add(self.label_content, 0, wx.ALL | wx.CENTER, 5)

        # Input fields for the wizard
        self.input_field = wx.TextCtrl(self)
        self.sizer.Add(self.input_field, 0, wx.ALL | wx.EXPAND, 5)

        self.SetSizer(self.sizer)

class Wizard(wx.Frame):
    def __init__(self):
        super(Wizard, self).__init__(None, title="Cool Wizard", size=(400, 300))
        
        self.pages = [
            WizardPage(self, "Step 1: Welcome!", "Please enter your name:"),
            WizardPage(self, "Step 2: Your Details", "Please enter your email:"),
            WizardPage(self, "Step 3: Confirmation", "Review your information and finish!")
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
        
        # Progress Bar
        self.progress = wx.Gauge(self, range=len(self.pages), size=(300, 25))
        self.sizer.Add(self.progress, 0, wx.ALL | wx.CENTER, 5)
        
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
        name = self.pages[0].input_field.GetValue()
        email = self.pages[1].input_field.GetValue()
        wx.MessageBox(f"Thank you, {name}!\nYour email: {email}", "Info", wx.OK | wx.ICON_INFORMATION)
        self.Close()
    
    def update_wizard(self):
        self.sizer.Hide(self.pages[self.current_page - 1])
        self.sizer.Show(self.pages[self.current_page])
        self.Layout()
        self.update_buttons()
        self.progress.SetValue(self.current_page + 1)
    
    def update_buttons(self):
        self.prev_button.Enable(self.current_page > 0)
        self.next_button.Enable(self.current_page < len(self.pages) - 1)
        self.finish_button.Enable(self.current_page == len(self.pages) - 1)

if __name__ == "__main__":
    app = wx.App(False)
    wizard = Wizard()
    wizard.Show()
    app.MainLoop()