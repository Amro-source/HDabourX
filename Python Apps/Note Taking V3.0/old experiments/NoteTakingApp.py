import wx
import sqlite3

class TreeApp(wx.Frame):
    def __init__(self, parent, title):
        super(TreeApp, self).__init__(parent, title=title, size=(800, 600))

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
        self.notes_list.InsertColumn(2, "Created At", width=150)
        self.notes_list.InsertColumn(3, "Updated At", width=150)
        
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

        # Show the frame
        self.Centre()
        self.Show()

    def populate_tree(self):
        """Populate the tree with tags from the database."""
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
        """Add a child node to the selected tree item and save it in the database."""
        item = self.tree.GetSelection()
        if item and item.IsOk():
            new_label = self.input_field.GetValue().strip()
            if new_label:
                parent_text = self.tree.GetItemText(item)
                conn = sqlite3.connect("notes.db")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO Tags (name, parent_id) VALUES (?, (SELECT id FROM Tags WHERE name = ?))", 
                               (new_label, parent_text))
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
