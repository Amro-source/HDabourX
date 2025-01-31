import wx
import sqlite3
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
        main_sizer.Add(wx.StaticText(self.panel, label="Description:"), 0, wx.ALL, 5)
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

class TreeApp(wx.Frame):
    def __init__(self, parent, title):
        super(TreeApp, self).__init__(parent, title=title, size=(800, 600))

        # Setup database
        self.setup_database()

        # Panel to hold the notebook
        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)

        # Tree Panel
        tree_panel = wx.Panel(notebook)
        self.tree = wx.TreeCtrl(tree_panel, style=wx.TR_DEFAULT_STYLE | wx.TR_HAS_BUTTONS)
        self.root = self.tree.AddRoot("Tags")
        self.populate_tree()
        self.tree.Expand(self.root)

        # Input field and button for adding child nodes
        self.input_field = wx.TextCtrl(tree_panel)
        self.add_button = wx.Button(tree_panel, label="Add Child")
        self.add_button.Bind(wx.EVT_BUTTON, self.on_add_child)

        # Layout for tree panel
        tree_sizer = wx.BoxSizer(wx.VERTICAL)
        tree_sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 10)
        tree_sizer.Add(self.input_field, 0, wx.EXPAND | wx.ALL, 5)
        tree_sizer.Add(self.add_button, 0, wx.EXPAND | wx.ALL, 5)
        tree_panel.SetSizer(tree_sizer)

        # Notes Panel
        notes_panel = wx.Panel(notebook)
        notes_sizer = wx.BoxSizer(wx.VERTICAL)

        self.notes_list = wx.ListCtrl(notes_panel, style=wx.LC_REPORT)
        self.notes_list.InsertColumn(0, "ID", width=50)
        self.notes_list.InsertColumn(1, "Title", width=200)
        self.notes_list.InsertColumn(2, "Content", width=300)  # Added this line
        self.notes_list.InsertColumn(3, "Created At", width=150)
        self.notes_list.InsertColumn(4, "Updated At", width=150)
        
        # Add buttons for CRUD operations
        self.add_note_btn = wx.Button(notes_panel, label="Add Note")
        self.edit_note_btn = wx.Button(notes_panel, label="Edit Note")
        self.delete_note_btn = wx.Button(notes_panel, label="Delete Note")

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.add_note_btn, 1, wx.EXPAND | wx.ALL, 5)
        btn_sizer.Add(self.edit_note_btn, 1, wx.EXPAND | wx.ALL, 5)
        btn_sizer.Add(self.delete_note_btn, 1, wx.EXPAND | wx.ALL, 5)

        notes_sizer.Add(self.notes_list, 1, wx.EXPAND | wx.ALL, 5)
        notes_sizer.Add(btn_sizer, 0, wx.EXPAND)
        notes_panel.SetSizer(notes_sizer)

        # Add panels to notebook
        notebook.AddPage(notes_panel, "Notes")
        notebook.AddPage(tree_panel, "Tags")

        # Main Layout
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(main_sizer)

        # Bind events
        self.add_note_btn.Bind(wx.EVT_BUTTON, self.on_add_note)
        self.edit_note_btn.Bind(wx.EVT_BUTTON, self.on_edit_note)
        self.delete_note_btn.Bind(wx.EVT_BUTTON, self.on_delete_note)

        # Show the frame
        self.Centre()
        self.Show()
        self.refresh_notes()

    def setup_database(self):
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        conn.close()

    def on_add_note(self, event):
        """Show a custom dialog for entering a note with title and description."""
        dialog = NoteDialog(self, title="Add Note")
        if dialog.ShowModal() == wx.ID_OK:
            data = dialog.GetNoteData()
            if data["title"].strip() and data["content"].strip():
                conn = sqlite3.connect("notes.db")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO Notes (title, content) VALUES (?, ?)", (data["title"], data["content"]))
                conn.commit()
                conn.close()
                self.refresh_notes()
        dialog.Destroy()


    def on_edit_note(self, event):
        selected_note = self.notes_list.GetFirstSelected()
        if selected_note != -1:
            note_id = self.notes_list.GetItemText(selected_note, col=0)
            dialog = wx.TextEntryDialog(self, "Edit Note Title:", "Edit Note", self.notes_list.GetItemText(selected_note, col=1))
            if dialog.ShowModal() == wx.ID_OK:
                new_title = dialog.GetValue()
                conn = sqlite3.connect("notes.db")
                cursor = conn.cursor()
                cursor.execute("UPDATE Notes SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_title, note_id))
                conn.commit()
                conn.close()
                self.refresh_notes()
            dialog.Destroy()

    def on_delete_note(self, event):
        selected_note = self.notes_list.GetFirstSelected()
        if selected_note != -1:
            note_id = self.notes_list.GetItemText(selected_note, col=0)
            conn = sqlite3.connect("notes.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Notes WHERE id = ?", (note_id,))
            conn.commit()
            conn.close()
            self.refresh_notes()

    def refresh_notes(self):
        """Refresh the notes list from the database."""
        self.notes_list.DeleteAllItems()
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content, created_at, updated_at FROM Notes")
        for row in cursor.fetchall():
            self.notes_list.Append([str(row[0]), row[1], row[2], row[3], row[4]])
        conn.close()


    def populate_tree(self):
        conn = sqlite3.connect("notes.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, parent_id FROM Tags")
        tags = cursor.fetchall()
        conn.close()
        tag_map = {None: self.root}
        for tag_id, tag_name, parent_id in tags:
            parent_item = tag_map.get(parent_id, self.root)
            tag_map[tag_id] = self.tree.AppendItem(parent_item, tag_name)

    def on_add_child(self, event):
        item = self.tree.GetSelection()
        if item and item.IsOk():
            new_label = self.input_field.GetValue().strip()
            if new_label:
                parent_text = self.tree.GetItemText(item)
                conn = sqlite3.connect("notes.db")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO Tags (name, parent_id) VALUES (?, (SELECT id FROM Tags WHERE name = ?))", (new_label, parent_text))
                conn.commit()
                conn.close()
                self.tree.AppendItem(item, new_label)
                self.input_field.Clear()
                self.tree.Expand(item)
            else:
                wx.MessageBox("Please enter a valid name.", "Error", wx.OK | wx.ICON_ERROR)
        else:
            wx.MessageBox("Please select a parent node.", "Error", wx.OK | wx.ICON_ERROR)

if __name__ == "__main__":
    app = wx.App(False)
    frame = TreeApp(None, "Notes & Tags Manager")
    app.MainLoop()
