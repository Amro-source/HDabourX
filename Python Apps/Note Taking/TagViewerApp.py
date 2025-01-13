import sqlite3
import wx

class TableViewer(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SetTitle("Database Table Viewer")
        self.SetSize((800, 600))

        # Notebook for tabs
        notebook = wx.Notebook(self)

        # Tabs for each table
        self.notes_tab = wx.Panel(notebook)
        self.tags_tab = wx.Panel(notebook)
        self.note_tags_tab = wx.Panel(notebook)

        notebook.AddPage(self.notes_tab, "Notes")
        notebook.AddPage(self.tags_tab, "Tags")
        notebook.AddPage(self.note_tags_tab, "NoteTags")

        # Set up each tab
        self.setup_notes_tab()
        self.setup_tags_tab()
        self.setup_note_tags_tab()

    def setup_notes_tab(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Notes ListCtrl
        self.notes_list = wx.ListCtrl(self.notes_tab, style=wx.LC_REPORT)
        self.notes_list.InsertColumn(0, "ID", width=50)
        self.notes_list.InsertColumn(1, "Title", width=200)
        self.notes_list.InsertColumn(2, "Content", width=300)
        self.notes_list.InsertColumn(3, "Created At", width=150)
        self.notes_list.InsertColumn(4, "Updated At", width=150)

        refresh_button = wx.Button(self.notes_tab, label="Refresh Notes")
        refresh_button.Bind(wx.EVT_BUTTON, self.refresh_notes)

        sizer.Add(self.notes_list, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(refresh_button, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        self.notes_tab.SetSizer(sizer)
        self.refresh_notes()

    def setup_tags_tab(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Tags ListCtrl
        self.tags_list = wx.ListCtrl(self.tags_tab, style=wx.LC_REPORT)
        self.tags_list.InsertColumn(0, "ID", width=50)
        self.tags_list.InsertColumn(1, "Name", width=300)
        self.tags_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_tag_selected)

        # Notes under selected tag
        self.notes_under_tag_list = wx.ListCtrl(self.tags_tab, style=wx.LC_REPORT)
        self.notes_under_tag_list.InsertColumn(0, "ID", width=50)
        self.notes_under_tag_list.InsertColumn(1, "Title", width=200)
        self.notes_under_tag_list.InsertColumn(2, "Content", width=300)

        refresh_button = wx.Button(self.tags_tab, label="Refresh Tags")
        refresh_button.Bind(wx.EVT_BUTTON, self.refresh_tags)

        sizer.Add(self.tags_list, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(wx.StaticText(self.tags_tab, label="Notes under selected tag:"), 0, wx.ALL, 5)
        sizer.Add(self.notes_under_tag_list, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(refresh_button, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        self.tags_tab.SetSizer(sizer)
        self.refresh_tags()

    def setup_note_tags_tab(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # NoteTags ListCtrl
        self.note_tags_list = wx.ListCtrl(self.note_tags_tab, style=wx.LC_REPORT)
        self.note_tags_list.InsertColumn(0, "Note ID", width=100)
        self.note_tags_list.InsertColumn(1, "Tag ID", width=100)

        refresh_button = wx.Button(self.note_tags_tab, label="Refresh NoteTags")
        refresh_button.Bind(wx.EVT_BUTTON, self.refresh_note_tags)

        sizer.Add(self.note_tags_list, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(refresh_button, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        self.note_tags_tab.SetSizer(sizer)
        self.refresh_note_tags()

    def refresh_notes(self, event=None):
        self.notes_list.DeleteAllItems()
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content, created_at, updated_at FROM Notes")
        for row in cursor.fetchall():
            self.notes_list.Append([str(row[0]), row[1], row[2], row[3], row[4]])
        conn.close()

    def refresh_tags(self, event=None):
        self.tags_list.DeleteAllItems()
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM Tags")
        for row in cursor.fetchall():
            self.tags_list.Append([str(row[0]), row[1]])
        conn.close()

    def refresh_note_tags(self, event=None):
        self.note_tags_list.DeleteAllItems()
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT note_id, tag_id FROM NoteTags")
        for row in cursor.fetchall():
            self.note_tags_list.Append([str(row[0]), str(row[1])])
        conn.close()

    def on_tag_selected(self, event):
        self.notes_under_tag_list.DeleteAllItems()
        selected_tag_index = event.GetIndex()
        tag_id = self.tags_list.GetItemText(selected_tag_index, col=0)

        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute('''
        SELECT Notes.id, Notes.title, Notes.content 
        FROM Notes
        JOIN NoteTags ON Notes.id = NoteTags.note_id
        WHERE NoteTags.tag_id = ?
        ''', (tag_id,))

        for row in cursor.fetchall():
            self.notes_under_tag_list.Append([str(row[0]), row[1], row[2]])

        conn.close()

if __name__ == "__main__":
    app = wx.App()
    frame = TableViewer(None)
    frame.Show()
    app.MainLoop()
