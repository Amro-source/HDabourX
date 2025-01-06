import wx
import wx.grid
import sqlite3
import csv
from datetime import datetime, timedelta


def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect("stopwatch.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY,
            description TEXT,
            start_time TEXT,
            end_time TEXT,
            duration TEXT
        )
    """)
    conn.commit()
    conn.close()


class StopwatchApp(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SetTitle("Stopwatch Tracker")
        self.SetSize((1000, 700))  # Larger default window size

        self.stopwatches = {}  # Dictionary to manage active stopwatches
        self.duration_labels = {}  # Dictionary to store duration labels for each stopwatch
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_durations, self.timer)
        self.timer.Start(1000)  # Update every second

        # Use a scrolled window for the main content
        self.scroll_panel = wx.ScrolledWindow(self, style=wx.VSCROLL | wx.HSCROLL)
        self.scroll_panel.SetScrollRate(10, 10)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Add controls
        self.description_input = wx.TextCtrl(self.scroll_panel, style=wx.TE_PROCESS_ENTER)
        self.description_input.SetHint("Enter activity description")
        self.start_button = wx.Button(self.scroll_panel, label="Start New Stopwatch")
        self.export_button = wx.Button(self.scroll_panel, label="Export to CSV")
        self.view_data_button = wx.Button(self.scroll_panel, label="View Historical Data")  # New Button

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header Row
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        header_sizer.Add(wx.StaticText(self.scroll_panel, label="ID", style=wx.ALIGN_CENTER), flag=wx.EXPAND, proportion=1)
        header_sizer.Add(wx.StaticText(self.scroll_panel, label="Description", style=wx.ALIGN_CENTER), flag=wx.EXPAND, proportion=2)
        header_sizer.Add(wx.StaticText(self.scroll_panel, label="Status", style=wx.ALIGN_CENTER), flag=wx.EXPAND, proportion=1)
        header_sizer.Add(wx.StaticText(self.scroll_panel, label="Duration", style=wx.ALIGN_CENTER), flag=wx.EXPAND, proportion=1)
        header_sizer.Add(wx.StaticText(self.scroll_panel, label="Actions", style=wx.ALIGN_CENTER), flag=wx.EXPAND, proportion=2)
        self.main_sizer.Add(header_sizer, flag=wx.EXPAND | wx.ALL, border=5)

        self.stopwatch_sizer = wx.BoxSizer(wx.VERTICAL)
        self.main_sizer.Add(self.stopwatch_sizer, proportion=1, flag=wx.EXPAND)

        # Bind events
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start_stopwatch)
        self.export_button.Bind(wx.EVT_BUTTON, self.export_to_csv)
        self.view_data_button.Bind(wx.EVT_BUTTON, self.view_historical_data)  # New Event Binding
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Add to layout
        vbox.Add(self.description_input, flag=wx.EXPAND | wx.ALL, border=5)
        vbox.Add(self.start_button, flag=wx.EXPAND | wx.ALL, border=5)
        vbox.Add(self.export_button, flag=wx.EXPAND | wx.ALL, border=5)
        vbox.Add(self.view_data_button, flag=wx.EXPAND | wx.ALL, border=5)  # New Button in Layout
        vbox.Add(self.main_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        self.scroll_panel.SetSizer(vbox)
        self.Show()

        init_db()

    def update_durations(self, event):
        """Update the duration labels for all running stopwatches."""
        for stopwatch_id, stopwatch in self.stopwatches.items():
            if stopwatch["status"] == "Running":
                duration = datetime.now() - stopwatch["start_time"]
                self.duration_labels[stopwatch_id].SetLabel(str(duration).split(".")[0])

    def on_start_stopwatch(self, event):
        description = self.description_input.GetValue().strip()
        if not description:
            wx.MessageBox("Please enter a description", "Error", wx.OK | wx.ICON_ERROR)
            return

        stopwatch_id = len(self.stopwatches) + 1
        start_time = datetime.now()

        # Add to active stopwatches
        self.stopwatches[stopwatch_id] = {
            "description": description,
            "start_time": start_time,
            "paused_time": None,
            "status": "Running",
            "duration": timedelta(0)
        }

        # Add to UI
        row_sizer = wx.BoxSizer(wx.HORIZONTAL)

        row_sizer.Add(wx.StaticText(self.scroll_panel, label=str(stopwatch_id), style=wx.ALIGN_CENTER), flag=wx.EXPAND, proportion=1)
        row_sizer.Add(wx.StaticText(self.scroll_panel, label=description, style=wx.ALIGN_CENTER), flag=wx.EXPAND, proportion=2)
        status_label = wx.StaticText(self.scroll_panel, label="Running", style=wx.ALIGN_CENTER)
        row_sizer.Add(status_label, flag=wx.EXPAND, proportion=1)
        duration_label = wx.StaticText(self.scroll_panel, label="0:00:00", style=wx.ALIGN_CENTER)
        row_sizer.Add(duration_label, flag=wx.EXPAND, proportion=1)

        # Store the label for updating
        self.duration_labels[stopwatch_id] = duration_label

        # Buttons
        actions_panel = wx.Panel(self.scroll_panel)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pause_button = wx.Button(actions_panel, label="Pause", size=(50, 25))
        resume_button = wx.Button(actions_panel, label="Resume", size=(60, 25))
        stop_button = wx.Button(actions_panel, label="Stop", size=(50, 25))
        resume_button.Enable(False)  # Disable resume button initially

        button_sizer.Add(pause_button, flag=wx.RIGHT, border=5)
        button_sizer.Add(resume_button, flag=wx.RIGHT, border=5)
        button_sizer.Add(stop_button, flag=wx.RIGHT, border=5)
        actions_panel.SetSizer(button_sizer)

        row_sizer.Add(actions_panel, flag=wx.EXPAND, proportion=2)

        self.stopwatch_sizer.Add(row_sizer, flag=wx.EXPAND | wx.ALL, border=5)
        self.stopwatch_sizer.Layout()
        self.scroll_panel.Layout()  # Update scrollable area

        # Bind button events
        pause_button.Bind(wx.EVT_BUTTON, lambda evt, sid=stopwatch_id: self.on_pause(evt, sid, pause_button, resume_button, status_label))
        resume_button.Bind(wx.EVT_BUTTON, lambda evt, sid=stopwatch_id: self.on_resume(evt, sid, pause_button, resume_button, status_label))
        stop_button.Bind(wx.EVT_BUTTON, lambda evt, sid=stopwatch_id: self.on_stop(evt, sid, status_label, duration_label))

        # Clear input
        self.description_input.SetValue("")

    def on_pause(self, event, stopwatch_id, pause_button, resume_button, status_label):
        stopwatch = self.stopwatches.get(stopwatch_id)
        if stopwatch and stopwatch["status"] == "Running":
            stopwatch["paused_time"] = datetime.now()
            stopwatch["status"] = "Paused"
            status_label.SetLabel("Paused")
            pause_button.Enable(False)
            resume_button.Enable(True)

    def on_resume(self, event, stopwatch_id, pause_button, resume_button, status_label):
        stopwatch = self.stopwatches.get(stopwatch_id)
        if stopwatch and stopwatch["status"] == "Paused":
            paused_duration = datetime.now() - stopwatch["paused_time"]
            stopwatch["start_time"] += paused_duration
            stopwatch["paused_time"] = None
            stopwatch["status"] = "Running"
            status_label.SetLabel("Running")
            pause_button.Enable(True)
            resume_button.Enable(False)

    def on_stop(self, event, stopwatch_id, status_label, duration_label):
        stopwatch = self.stopwatches.get(stopwatch_id)
        if stopwatch:
            end_time = datetime.now()
            duration = end_time - stopwatch["start_time"]
            stopwatch["duration"] = duration
            stopwatch["status"] = "Stopped"

            # Save to DB
            conn = sqlite3.connect("stopwatch.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO activities (description, start_time, end_time, duration)
                VALUES (?, ?, ?, ?)
            """, (stopwatch["description"], stopwatch["start_time"], end_time, str(duration)))
            conn.commit()
            conn.close()

            status_label.SetLabel("Stopped")
            duration_label.SetLabel(str(duration).split(".")[0])
            del self.stopwatches[stopwatch_id]

    def export_to_csv(self, event):
        with open("stopwatch_data.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Description", "Start Time", "End Time", "Duration"])

            conn = sqlite3.connect("stopwatch.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM activities")
            for row in cursor.fetchall():
                writer.writerow(row)
            conn.close()

        wx.MessageBox("Data exported to stopwatch_data.csv", "Export Successful", wx.OK | wx.ICON_INFORMATION)

    def view_historical_data(self, event):
        """Fetch and display historical data from the SQLite database."""
        conn = sqlite3.connect("stopwatch.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM activities")
        rows = cursor.fetchall()
        conn.close()

        # Create a new window to display data
        data_frame = wx.Frame(self, title="Historical Data", size=(800, 600))
        panel = wx.Panel(data_frame)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Add a grid for data display
        grid = wx.grid.Grid(panel)
        grid.CreateGrid(len(rows), 5)  # Number of rows and columns
        grid.SetColLabelValue(0, "ID")
        grid.SetColLabelValue(1, "Description")
        grid.SetColLabelValue(2, "Start Time")
        grid.SetColLabelValue(3, "End Time")
        grid.SetColLabelValue(4, "Duration")

        # Populate grid with data
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                grid.SetCellValue(i, j, str(value))

        vbox.Add(grid, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(vbox)
        data_frame.Show()

    def on_close(self, event):
        self.Destroy()


# Run the application
if __name__ == "__main__":
    app = wx.App(False)
    frame = StopwatchApp(None)
    app.MainLoop()
