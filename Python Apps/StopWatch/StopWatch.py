import wx
import wx.adv
import time
from threading import Thread
from datetime import datetime, timedelta
from playsound import playsound  # For alarm sound


class ClockApp(wx.Frame):
    def __init__(self, *args, **kw):
        super(ClockApp, self).__init__(*args, **kw)

        self.InitUI()
        self.SetSize((500, 400))
        self.SetTitle("Clock, Alarm, Timer, Stopwatch")
        self.Centre()

        # Timer, stopwatch, and alarm variables
        self.timer_running = False
        self.stopwatch_running = False
        self.stopwatch_start_time = None
        self.alarm_list = []

    def InitUI(self):
        panel = wx.Panel(self)

        # Notebook for tabs
        notebook = wx.Notebook(panel)
        self.clock_panel = wx.Panel(notebook)
        self.alarm_panel = wx.Panel(notebook)
        self.timer_panel = wx.Panel(notebook)
        self.stopwatch_panel = wx.Panel(notebook)

        notebook.AddPage(self.clock_panel, "Clock")
        notebook.AddPage(self.alarm_panel, "Alarm")
        notebook.AddPage(self.timer_panel, "Timer")
        notebook.AddPage(self.stopwatch_panel, "Stopwatch")

        # CLOCK PANEL
        self.clock_label = wx.StaticText(self.clock_panel, label="", style=wx.ALIGN_CENTER)
        font = wx.Font(20, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.clock_label.SetFont(font)
        self.update_clock()
        clock_sizer = wx.BoxSizer(wx.VERTICAL)
        clock_sizer.Add(self.clock_label, 1, wx.ALIGN_CENTER | wx.ALL, 20)
        self.clock_panel.SetSizer(clock_sizer)

        # ALARM PANEL
        alarm_sizer = wx.BoxSizer(wx.VERTICAL)
        self.alarm_picker = wx.adv.TimePickerCtrl(self.alarm_panel)
        self.add_alarm_btn = wx.Button(self.alarm_panel, label="Add Alarm")
        self.alarm_listbox = wx.ListBox(self.alarm_panel, style=wx.LB_SINGLE)

        self.add_alarm_btn.Bind(wx.EVT_BUTTON, self.add_alarm)
        self.alarm_listbox.Bind(wx.EVT_LISTBOX_DCLICK, self.remove_alarm)

        alarm_sizer.Add(wx.StaticText(self.alarm_panel, label="Set Alarm:"), 0, wx.ALL, 10)
        alarm_sizer.Add(self.alarm_picker, 0, wx.EXPAND | wx.ALL, 10)
        alarm_sizer.Add(self.add_alarm_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        alarm_sizer.Add(wx.StaticText(self.alarm_panel, label="Active Alarms:"), 0, wx.ALL, 10)
        alarm_sizer.Add(self.alarm_listbox, 1, wx.EXPAND | wx.ALL, 10)

        self.alarm_panel.SetSizer(alarm_sizer)

        # TIMER PANEL
        timer_sizer = wx.BoxSizer(wx.VERTICAL)
        self.timer_input = wx.SpinCtrl(self.timer_panel, min=1, max=3600, initial=60)  # Timer in seconds
        self.timer_label = wx.StaticText(self.timer_panel, label="", style=wx.ALIGN_CENTER)
        self.start_timer_btn = wx.Button(self.timer_panel, label="Start Timer")
        self.stop_timer_btn = wx.Button(self.timer_panel, label="Stop Timer")

        self.start_timer_btn.Bind(wx.EVT_BUTTON, self.start_timer)
        self.stop_timer_btn.Bind(wx.EVT_BUTTON, self.stop_timer)

        timer_sizer.Add(wx.StaticText(self.timer_panel, label="Set Timer (seconds):"), 0, wx.ALL, 10)
        timer_sizer.Add(self.timer_input, 0, wx.ALL, 10)
        timer_sizer.Add(self.timer_label, 0, wx.EXPAND | wx.ALL, 10)
        timer_sizer.Add(self.start_timer_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        timer_sizer.Add(self.stop_timer_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        self.timer_panel.SetSizer(timer_sizer)

        # STOPWATCH PANEL
        stopwatch_sizer = wx.BoxSizer(wx.VERTICAL)
        self.stopwatch_label = wx.StaticText(self.stopwatch_panel, label="00:00:00", style=wx.ALIGN_CENTER)
        self.start_stopwatch_btn = wx.Button(self.stopwatch_panel, label="Start")
        self.stop_stopwatch_btn = wx.Button(self.stopwatch_panel, label="Stop")
        self.reset_stopwatch_btn = wx.Button(self.stopwatch_panel, label="Reset")

        self.start_stopwatch_btn.Bind(wx.EVT_BUTTON, self.start_stopwatch)
        self.stop_stopwatch_btn.Bind(wx.EVT_BUTTON, self.stop_stopwatch)
        self.reset_stopwatch_btn.Bind(wx.EVT_BUTTON, self.reset_stopwatch)

        stopwatch_sizer.Add(self.stopwatch_label, 0, wx.EXPAND | wx.ALL, 10)
        stopwatch_sizer.Add(self.start_stopwatch_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        stopwatch_sizer.Add(self.stop_stopwatch_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        stopwatch_sizer.Add(self.reset_stopwatch_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        self.stopwatch_panel.SetSizer(stopwatch_sizer)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.EXPAND)
        panel.SetSizer(sizer)

    def update_clock(self):
        """Update clock display every second."""
        self.clock_label.SetLabel(datetime.now().strftime("%H:%M:%S"))
        wx.CallLater(1000, self.update_clock)

    def add_alarm(self, event):
        """Add an alarm to the list."""
        alarm_time = self.alarm_picker.GetValue()
        alarm_time_str = alarm_time.Format("%H:%M:%S")
        self.alarm_list.append(alarm_time_str)
        self.alarm_listbox.Append(alarm_time_str)
        Thread(target=self.check_alarm, args=(alarm_time_str,)).start()

    def remove_alarm(self, event):
        """Remove an alarm when double-clicked."""
        selection = self.alarm_listbox.GetSelection()
        if selection != wx.NOT_FOUND:
            self.alarm_list.pop(selection)
            self.alarm_listbox.Delete(selection)

    def check_alarm(self, alarm_time_str):
        """Check if the current time matches any alarm."""
        while alarm_time_str in self.alarm_list:
            now = datetime.now().strftime("%H:%M:%S")
            if now >= alarm_time_str:
                wx.CallAfter(wx.MessageBox, "Alarm ringing!", "Alarm", wx.OK | wx.ICON_INFORMATION)
                wx.CallAfter(self.remove_alarm_by_time, alarm_time_str)
                # Play the alarm sound
                playsound("alarm.mp3")  # Replace with your sound file
                break
            time.sleep(1)

    def remove_alarm_by_time(self, alarm_time_str):
        """Remove an alarm from the list by time."""
        if alarm_time_str in self.alarm_list:
            index = self.alarm_list.index(alarm_time_str)
            self.alarm_list.pop(index)
            self.alarm_listbox.Delete(index)

    def start_timer(self, event):
        """Start a countdown timer."""
        duration = self.timer_input.GetValue()
        end_time = datetime.now() + timedelta(seconds=duration)
        self.timer_running = True
        Thread(target=self.run_timer, args=(end_time,)).start()

    def run_timer(self, end_time):
        """Run the countdown timer."""
        while self.timer_running and datetime.now() < end_time:
            remaining = end_time - datetime.now()
            self.timer_label.SetLabel(f"Remaining: {str(remaining).split('.')[0]}")
            time.sleep(1)
        if self.timer_running:
            wx.CallAfter(wx.MessageBox, "Timer finished!", "Timer", wx.OK | wx.ICON_INFORMATION)
        self.timer_running = False

    def stop_timer(self, event):
        """Stop the countdown timer."""
        self.timer_running = False

    def start_stopwatch(self, event):
        """Start the stopwatch."""
        if not self.stopwatch_running:
            self.stopwatch_running = True
            self.stopwatch_start_time = datetime.now()
            Thread(target=self.run_stopwatch).start()

    def run_stopwatch(self):
        """Run the stopwatch."""
        while self.stopwatch_running:
            elapsed = datetime.now() - self.stopwatch_start_time
            self.stopwatch_label.SetLabel(str(elapsed).split(".")[0])
            time.sleep(0.1)

    def stop_stopwatch(self, event):
        """Stop the stopwatch."""
        self.stopwatch_running = False

    def reset_stopwatch(self, event):
        """Reset the stopwatch."""
        self.stopwatch_running = False
        self.stopwatch_label.SetLabel("00:00:00")


if __name__ == '__main__':
    app = wx.App(False)
    frame = ClockApp(None)
    frame.Show()
    app.MainLoop()
