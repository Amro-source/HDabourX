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
        name TEXT UNIQUE NOT NULL,
        parent_id INTEGER,
        FOREIGN KEY(parent_id) REFERENCES Tags(id) ON DELETE SET NULL
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

    def SetNoteData(self, title, content):
        self.title_ctrl.SetValue(title)
        self.content_ctrl.SetValue(content)

# Tag Dialog Class with Parent Selection
class TagDialog(wx.Dialog):
    def __init__(self, parent, title="Add Tag"):
        super().__init__(parent, title=title, size=(400, 300))

        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Tag Name Input
        main_sizer.Add(wx.StaticText(self.panel, label="Tag Name:"), 0, wx.ALL, 5)
        self.tag_ctrl = wx.TextCtrl(self.panel)
        main_sizer.Add(self.tag_ctrl, 0, wx.EXPAND | wx.ALL, 5)

        # Parent Tag Selection
        main_sizer.Add(wx.StaticText(self.panel, label="Parent Tag (Optional):"), 0, wx.ALL, 5)
        self.parent_choice = wx.Choice(self.panel, choices=[])
        main_sizer.Add(self.parent_choice, 0, wx.EXPAND | wx.ALL, 5)

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

    def PopulateParentTags(self):
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM Tags")
        tags = cursor.fetchall()
        conn.close()

        self.parent_choice.Clear()
        self.parent_choice.Append("None", -1)
        for tag_id, tag_name in tags:
            self.parent_choice.Append(tag_name, tag_id)
        self.parent_choice.Select(0)

    def OnSave(self, event):
        self.EndModal(wx.ID_OK)

    def OnCancel(self, event):
        self.EndModal(wx.ID_CANCEL)

    def GetTagData(self):
        return {
            "name": self.tag_ctrl.GetValue(),
            "parent_id": self.parent_choice.GetClientData(self.parent_choice.GetSelection())
        }

# BlogApp GUI Updates
class BlogApp(wx.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SetTitle("Notes & Tags Manager with Hierarchical Tags")
        self.SetSize((1000, 600))

        # Layout
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Notes List Panel
        self.notes_list = wx.ListCtrl(self.panel, style=wx.LC_REPORT)
        self.notes_list.InsertColumn(0, "ID", width=50)
        self.notes_list.InsertColumn(1, "Title", width=200)
        self.notes_list.InsertColumn(2, "Created At", width=150)
        self.notes_list.InsertColumn(3, "Updated At", width=150)

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
        self.tree = wx.TreeCtrl(self.panel)
        self.add_tag_btn = wx.Button(self.panel, label="Add Tag")

        tags_sizer = wx.BoxSizer(wx.VERTICAL)
        tags_sizer.Add(wx.StaticText(self.panel, label="Tags"), 0, wx.ALL, 5)
        tags_sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)
        tags_sizer.Add(self.add_tag_btn, 0, wx.EXPAND | wx.ALL, 5)

        # Combine into main layout
        main_sizer.Add(notes_sizer, 3, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(tags_sizer, 2, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(main_sizer)

        # Bind events
        self.BindEvents()

    def BindEvents(self):
        self.add_note_btn.Bind(wx.EVT_BUTTON, self.OnAddNote)
        self.edit_note_btn.Bind(wx.EVT_BUTTON, self.OnEditNote)
        self.delete_note_btn.Bind(wx.EVT_BUTTON, self.OnDeleteNote)
        self.add_tag_btn.Bind(wx.EVT_BUTTON, self.OnAddTag)
        self.notes_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnNoteSelected)

    def RefreshTagHierarchy(self):
        self.tree.DeleteAllItems()
        root = self.tree.AddRoot("Tags")

        def AddTagToTree(tag_id, tag_name, parent_item):
            child_item = self.tree.AppendItem(parent_item, tag_name)
            conn = sqlite3.connect("notes.db")
            cursor = conn.cursor()
            cursor.execute("SELECT id, name FROM Tags WHERE parent_id = ?", (tag_id,))
            for child_id, child_name in cursor.fetchall():
                AddTagToTree(child_id, child_name, child_item)
            conn.close()

        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM Tags WHERE parent_id IS NULL")
        for tag_id, tag_name in cursor.fetchall():
            AddTagToTree(tag_id, tag_name, root)
        conn.close()

        self.tree.Expand(root)

    def RefreshTags(self, note_id):
        self.tree.DeleteAllItems()
        root = self.tree.AddRoot("Tags for Note")
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute('''
        SELECT Tags.id, Tags.name FROM Tags
        JOIN NoteTags ON Tags.id = NoteTags.tag_id
        WHERE NoteTags.note_id = ?
        ''', (note_id,))
        for tag_id, tag_name in cursor.fetchall():
            self.tree.AppendItem(root, tag_name)
        conn.close()
        self.tree.Expand(root)

    def OnAddNote(self, event):
        dialog = NoteDialog(self, title="Add Note")
        if dialog.ShowModal() == wx.ID_OK:
            data = dialog.GetNoteData()
            self.SaveNoteToDatabase(data)
            self.RefreshNotes()
        dialog.Destroy()

    def SaveNoteToDatabase(self, data):
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Notes (title, content) VALUES (?, ?)", (data["title"], data["content"]))
        conn.commit()
        conn.close()

    def RefreshNotes(self):
        self.notes_list.DeleteAllItems()
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at, updated_at FROM Notes")
        for row in cursor.fetchall():
            self.notes_list.Append([str(row[0]), row[1], row[2], row[3]])
        conn.close()

    def OnEditNote(self, event):
        selected_note = self.notes_list.GetFirstSelected()
        if selected_note != -1:
            note_id = self.notes_list.GetItemText(selected_note, col=0)
            conn = sqlite3.connect("notes.db")
            cursor = conn.cursor()
            cursor.execute("SELECT title, content FROM Notes WHERE id = ?", (note_id,))
            note = cursor.fetchone()
            conn.close()

            if note:
                dialog = NoteDialog(self, title="Edit Note")
                dialog.SetNoteData(note[0], note[1])
                if dialog.ShowModal() == wx.ID_OK:
                    data = dialog.GetNoteData()
                    self.UpdateNoteInDatabase(note_id, data)
                    self.RefreshNotes()
                dialog.Destroy()

    def UpdateNoteInDatabase(self, note_id, data):
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE Notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (data["title"], data["content"], note_id))
        conn.commit()
        conn.close()

    def OnNoteSelected(self, event):
        selected_index = event.GetIndex()
        note_id = self.notes_list.GetItemText(selected_index, col=0)
        self.RefreshTags(note_id)

    def OnAddTag(self, event):
        selected_note = self.notes_list.GetFirstSelected()
        if selected_note == -1:
            wx.MessageBox("Please select a note to add a tag.", "Warning")
            return

        dialog = TagDialog(self, title="Add Tag")
        dialog.PopulateParentTags()
        if dialog.ShowModal() == wx.ID_OK:
            tag_data = dialog.GetTagData()
            note_id = self.notes_list.GetItemText(selected_note, col=0)
            self.SaveTagToDatabase(tag_data, note_id)
        dialog.Destroy()

    def SaveTagToDatabase(self, tag_data, note_id):
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        try:
            # Insert tag into Tags table
            cursor.execute("INSERT OR IGNORE INTO Tags (name, parent_id) VALUES (?, ?)",
                           (tag_data["name"], tag_data["parent_id"]))
            conn.commit()

            # Retrieve the tag ID
            cursor.execute("SELECT id FROM Tags WHERE name = ?", (tag_data["name"],))
            tag_id = cursor.fetchone()[0]

            # Link tag to the note
            cursor.execute("INSERT OR IGNORE INTO NoteTags (note_id, tag_id) VALUES (?, ?)", (note_id, tag_id))
            conn.commit()
            wx.MessageBox("Tag added successfully!", "Info")
        except sqlite3.IntegrityError as e:
            wx.MessageBox(f"Error adding tag: {str(e)}", "Error")
        conn.close()
        self.RefreshTags(note_id)

    def OnDeleteNote(self, event):
        selected_note = self.notes_list.GetFirstSelected()
        if selected_note != -1:
            note_id = self.notes_list.GetItemText(selected_note, col=0)
            conn = sqlite3.connect("notes.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Notes WHERE id = ?", (note_id,))
            conn.commit()
            conn.close()
            self.RefreshNotes()

if __name__ == "__main__":
    setup_database()
    app = wx.App()
    frame = BlogApp(None)
    frame.Show()
    app.MainLoop()
