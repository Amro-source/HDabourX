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

# Note Dialog Class
class NoteDialog(wx.Dialog):
    def __init__(self, parent, title="Add Note"):
        super().__init__(parent, title=title, size=(400, 300))

        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title Input
        main_sizer.Add(wx.StaticText(self.panel, label="Title:"), 0, wx.ALL, 5)
        self.title_ctrl = wx.TextCtrl(self.panel)
        main_sizer.Add(self.title_ctrl, 0, wx.EXPAND | wx.ALL, 5)

        # Content Input
        main_sizer.Add(wx.StaticText(self.panel, label="Content:"), 0, wx.ALL, 5)
        self.content_ctrl = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE)
        main_sizer.Add(self.content_ctrl, 1, wx.EXPAND | wx.ALL, 5)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        save_button = wx.Button(self.panel, label="Save")
        cancel_button = wx.Button(self.panel, label="Cancel")
        button_sizer.Add(save_button, 1, wx.EXPAND | wx.ALL, 5)
        button_sizer.Add(cancel_button, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)

        # Bind Events
        save_button.Bind(wx.EVT_BUTTON, self.OnSave)
        cancel_button.Bind(wx.EVT_BUTTON, self.OnCancel)

        self.panel.SetSizer(main_sizer)

    def OnSave(self, event):
        self.EndModal(wx.ID_OK)

    def OnCancel(self, event):
        self.EndModal(wx.ID_CANCEL)

    def GetNoteData(self):
        return {
            "title": self.title_ctrl.GetValue(),
            "content": self.content_ctrl.GetValue()
        }

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
        dialog = NoteDialog(self, title="Add Note")
        if dialog.ShowModal() == wx.ID_OK:
            data = dialog.GetNoteData()
            self.SaveNoteToDatabase(data)
        dialog.Destroy()

    def SaveNoteToDatabase(self, data):
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Notes (title, content) VALUES (?, ?)", (data["title"], data["content"]))
        conn.commit()
        conn.close()
        wx.MessageBox("Note added successfully!", "Info")

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
