import wx

class TabOne(wx.Panel):
    def __init__(self, parent, notebook):
        super(TabOne, self).__init__(parent)
        self.notebook = notebook

        # Create a sizer for layout
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Add a label
        label = wx.StaticText(self, label="Welcome to Tab 1!")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 5)

        # Add a button
        self.button = wx.Button(self, label="Go to Tab 2")
        self.button.Bind(wx.EVT_BUTTON, self.on_button_click)
        sizer.Add(self.button, 0, wx.ALL | wx.CENTER, 5)

        # Set the sizer
        self.SetSizer(sizer)

    def on_button_click(self, event):
        # Switch to Tab 2
        self.notebook.ChangeSelection(1)  # Index 1 corresponds to Tab 2
        print("Switched to Tab 2!")


class TabTwo(wx.Panel):
    def __init__(self, parent):
        super(TabTwo, self).__init__(parent)

        # Create a sizer for layout
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Add a label
        label = wx.StaticText(self, label="You are now in Tab 2!")
        sizer.Add(label, 0, wx.ALL | wx.CENTER, 5)

        # Add a cool gauge (progress bar)
        self.gauge = wx.Gauge(self, range=100, size=(250, 25))
        sizer.Add(self.gauge, 0, wx.ALL | wx.CENTER, 5)

        # Start a timer to simulate progress
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_gauge, self.timer)
        self.progress = 0
        self.timer.Start(100)  # Update every 100ms

        # Set the sizer
        self.SetSizer(sizer)

    def update_gauge(self, event):
        if self.progress >= 100:
            self.timer.Stop()
            return
        self.progress += 1
        self.gauge.SetValue(self.progress)


class MainFrame(wx.Frame):
    def __init__(self):
        super(MainFrame, self).__init__(None, title="wxPython Tabbed App", size=(400, 300))

        # Create a notebook (tabbed interface)
        self.notebook = wx.Notebook(self)

        # Add tabs
        self.tab_one = TabOne(self.notebook, self.notebook)
        self.tab_two = TabTwo(self.notebook)

        self.notebook.AddPage(self.tab_one, "Tab 1")
        self.notebook.AddPage(self.tab_two, "Tab 2")

        # Set the notebook as the main content
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(sizer)

        # Center the frame and show it
        self.Centre()
        self.Show()


if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()