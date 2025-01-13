import sqlite3
import wx

# Database Setup
def setup_database():
    conn = sqlite3.connect("notes.db")
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS NoteTags (
        note_id INTEGER,
        tag_id INTEGER,
        FOREIGN KEY(note_id) REFERENCES Notes(id) ON DELETE CASCADE,
        FOREIGN KEY(tag_id) REFERENCES Tags(id) ON DELETE CASCADE,
        PRIMARY KEY(note_id, tag_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TreeRelations (
        parent_id INTEGER,
        child_id INTEGER,
        FOREIGN KEY(parent_id) REFERENCES Notes(id) ON DELETE CASCADE,
        FOREIGN KEY(child_id) REFERENCES Notes(id) ON DELETE CASCADE,
        PRIMARY KEY(parent_id, child_id)
    )
    ''')

    conn.commit()
    conn.close()

# GUI Setup
class BlogApp(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SetTitle("Notes & Tags Manager")
        self.SetSize((1000, 600))

        # Layout
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Notes List Panel
        self.notes_list = wx.ListCtrl(self.panel, style=wx.LC_REPORT)
        self.notes_list.InsertColumn(0, "Title", width=200)
        self.notes_list.InsertColumn(1, "Created At", width=150)
        self.notes_list.InsertColumn(2, "Updated At", width=150)

        # Add buttons for CRUD operations
        self.add_note_btn = wx.Button(self.panel, label="Add Note")
        self.edit_note_btn = wx.Button(self.panel, label="Edit Note")
        self.delete_note_btn = wx.Button(self.panel, label="Delete Note")

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.add_note_btn, 1, wx.EXPAND | wx.ALL, 5)
        btn_sizer.Add(self.edit_note_btn, 1, wx.EXPAND | wx.ALL, 5)
        btn_sizer.Add(self.delete_note_btn, 1, wx.EXPAND | wx.ALL, 5)

        notes_sizer = wx.BoxSizer(wx.VERTICAL)
        notes_sizer.Add(self.notes_list, 1, wx.EXPAND | wx.ALL, 5)
        notes_sizer.Add(btn_sizer, 0, wx.EXPAND)

        # Tags Sidebar
        self.tags_list = wx.ListBox(self.panel)
        self.add_tag_btn = wx.Button(self.panel, label="Add Tag")
        self.delete_tag_btn = wx.Button(self.panel, label="Delete Tag")

        tags_sizer = wx.BoxSizer(wx.VERTICAL)
        tags_sizer.Add(wx.StaticText(self.panel, label="Tags"), 0, wx.ALL, 5)
        tags_sizer.Add(self.tags_list, 1, wx.EXPAND | wx.ALL, 5)
        tags_sizer.Add(self.add_tag_btn, 0, wx.EXPAND | wx.ALL, 5)
        tags_sizer.Add(self.delete_tag_btn, 0, wx.EXPAND | wx.ALL, 5)

        # Tree View Panel
        self.tree = wx.TreeCtrl(self.panel)
        tree_sizer = wx.BoxSizer(wx.VERTICAL)
        tree_sizer.Add(wx.StaticText(self.panel, label="Hierarchy"), 0, wx.ALL, 5)
        tree_sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)

        # Add to main sizer
        main_sizer.Add(notes_sizer, 3, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(tags_sizer, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(tree_sizer, 2, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(main_sizer)

        # Bind events
        self.BindEvents()

    def BindEvents(self):
        self.add_note_btn.Bind(wx.EVT_BUTTON, self.OnAddNote)
        self.edit_note_btn.Bind(wx.EVT_BUTTON, self.OnEditNote)
        self.delete_note_btn.Bind(wx.EVT_BUTTON, self.OnDeleteNote)
        self.add_tag_btn.Bind(wx.EVT_BUTTON, self.OnAddTag)
        self.delete_tag_btn.Bind(wx.EVT_BUTTON, self.OnDeleteTag)

    def OnAddNote(self, event):
        wx.MessageBox("Add Note functionality here", "Info")

    def OnEditNote(self, event):
        wx.MessageBox("Edit Note functionality here", "Info")

    def OnDeleteNote(self, event):
        wx.MessageBox("Delete Note functionality here", "Info")

    def OnAddTag(self, event):
        wx.MessageBox("Add Tag functionality here", "Info")

    def OnDeleteTag(self, event):
        wx.MessageBox("Delete Tag functionality here", "Info")

if __name__ == "__main__":
    setup_database()

    app = wx.App()
    frame = BlogApp(None)
    frame.Show()
    app.MainLoop()
