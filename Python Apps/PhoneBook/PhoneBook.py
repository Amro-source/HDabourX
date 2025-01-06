import wx
import sqlite3

class PhoneBookApp(wx.Frame):
    def __init__(self, parent, title):
        super(PhoneBookApp, self).__init__(parent, title=title, size=(700, 500))

        # Connect to SQLite database
        self.conn = sqlite3.connect('phonebook.db')
        self.create_table()

        # Main panel
        self.panel = wx.Panel(self)

        # Layout components
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Search bar
        search_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.search_box = wx.TextCtrl(self.panel, style=wx.TE_PROCESS_ENTER, size=(400, 25))
        search_button = wx.Button(self.panel, label="Search")
        search_button.Bind(wx.EVT_BUTTON, self.on_search)
        search_sizer.Add(self.search_box, 1, wx.EXPAND | wx.ALL, 5)
        search_sizer.Add(search_button, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(search_sizer, 0, wx.EXPAND)

        # List of contacts
        self.contact_list = wx.ListCtrl(self.panel, style=wx.LC_REPORT)
        self.contact_list.InsertColumn(0, "ID", width=50)
        self.contact_list.InsertColumn(1, "Name", width=150)
        self.contact_list.InsertColumn(2, "Phone", width=120)
        self.contact_list.InsertColumn(3, "Email", width=150)
        self.contact_list.InsertColumn(4, "Address", width=200)
        self.load_contacts()
        main_sizer.Add(self.contact_list, 1, wx.EXPAND | wx.ALL, 5)

        # Buttons for Add, Edit, Delete
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        add_button = wx.Button(self.panel, label="Add Contact")
        edit_button = wx.Button(self.panel, label="Edit Contact")
        delete_button = wx.Button(self.panel, label="Delete Contact")
        add_button.Bind(wx.EVT_BUTTON, self.on_add)
        edit_button.Bind(wx.EVT_BUTTON, self.on_edit)
        delete_button.Bind(wx.EVT_BUTTON, self.on_delete)
        button_sizer.Add(add_button, 0, wx.EXPAND | wx.ALL, 5)
        button_sizer.Add(edit_button, 0, wx.EXPAND | wx.ALL, 5)
        button_sizer.Add(delete_button, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(button_sizer, 0, wx.CENTER)

        # Set main sizer
        self.panel.SetSizer(main_sizer)

    def create_table(self):
        """Create the contacts table if it doesn't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    email TEXT,
                    address TEXT
                )
            """)

    def load_contacts(self):
        """Load contacts into the list control."""
        self.contact_list.DeleteAllItems()
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM contacts")
            for row in cursor:
                self.contact_list.Append(row)

    def on_search(self, event):
        """Search for contacts by name or phone."""
        query = self.search_box.GetValue()
        self.contact_list.DeleteAllItems()
        with self.conn:
            cursor = self.conn.execute("""
                SELECT * FROM contacts WHERE name LIKE ? OR phone LIKE ?
            """, (f'%{query}%', f'%{query}%'))
            for row in cursor:
                self.contact_list.Append(row)

    def on_add(self, event):
        """Add a new contact."""
        dialog = ContactDialog(self, title="Add Contact")
        if dialog.ShowModal() == wx.ID_OK:
            name, phone, email, address = dialog.get_data()
            with self.conn:
                self.conn.execute("""
                    INSERT INTO contacts (name, phone, email, address)
                    VALUES (?, ?, ?, ?)
                """, (name, phone, email, address))
            self.load_contacts()
        dialog.Destroy()

    def on_edit(self, event):
        """Edit the selected contact."""
        selected = self.contact_list.GetFirstSelected()
        if selected == -1:
            wx.MessageBox("No contact selected.", "Error", wx.OK | wx.ICON_ERROR)
            return

        contact_id = self.contact_list.GetItemText(selected)
        name = self.contact_list.GetItem(selected, 1).GetText()
        phone = self.contact_list.GetItem(selected, 2).GetText()
        email = self.contact_list.GetItem(selected, 3).GetText()
        address = self.contact_list.GetItem(selected, 4).GetText()

        dialog = ContactDialog(self, title="Edit Contact", name=name, phone=phone, email=email, address=address)
        if dialog.ShowModal() == wx.ID_OK:
            name, phone, email, address = dialog.get_data()
            with self.conn:
                self.conn.execute("""
                    UPDATE contacts SET name=?, phone=?, email=?, address=?
                    WHERE id=?
                """, (name, phone, email, address, contact_id))
            self.load_contacts()
        dialog.Destroy()

    def on_delete(self, event):
        """Delete the selected contact."""
        selected = self.contact_list.GetFirstSelected()
        if selected == -1:
            wx.MessageBox("No contact selected.", "Error", wx.OK | wx.ICON_ERROR)
            return

        contact_id = self.contact_list.GetItemText(selected)
        with self.conn:
            self.conn.execute("DELETE FROM contacts WHERE id=?", (contact_id,))
        self.load_contacts()


class ContactDialog(wx.Dialog):
    def __init__(self, parent, title="Contact Details", name='', phone='', email='', address=''):
        super(ContactDialog, self).__init__(parent, size=(400, 300))

        # Set the dialog title
        self.SetTitle(title)

        # Main sizer for the dialog
        dialog_sizer = wx.BoxSizer(wx.VERTICAL)

        # Create a panel for the dialog content
        self.panel = wx.Panel(self)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)

        # Name field
        panel_sizer.Add(wx.StaticText(self.panel, label="Name"), 0, wx.EXPAND | wx.ALL, 5)
        self.name_ctrl = wx.TextCtrl(self.panel, value=name)
        panel_sizer.Add(self.name_ctrl, 0, wx.EXPAND | wx.ALL, 5)

        # Phone field
        panel_sizer.Add(wx.StaticText(self.panel, label="Phone"), 0, wx.EXPAND | wx.ALL, 5)
        self.phone_ctrl = wx.TextCtrl(self.panel, value=phone)
        panel_sizer.Add(self.phone_ctrl, 0, wx.EXPAND | wx.ALL, 5)

        # Email field
        panel_sizer.Add(wx.StaticText(self.panel, label="Email"), 0, wx.EXPAND | wx.ALL, 5)
        self.email_ctrl = wx.TextCtrl(self.panel, value=email)
        panel_sizer.Add(self.email_ctrl, 0, wx.EXPAND | wx.ALL, 5)

        # Address field
        panel_sizer.Add(wx.StaticText(self.panel, label="Address"), 0, wx.EXPAND | wx.ALL, 5)
        self.address_ctrl = wx.TextCtrl(self.panel, value=address)
        panel_sizer.Add(self.address_ctrl, 0, wx.EXPAND | wx.ALL, 5)

        # Set the sizer for the panel
        self.panel.SetSizer(panel_sizer)

        # Add the panel to the dialog sizer
        dialog_sizer.Add(self.panel, 1, wx.EXPAND | wx.ALL, 5)

        # Add standard OK/Cancel buttons directly to the dialog
        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        dialog_sizer.Add(btn_sizer, 0, wx.CENTER | wx.ALL, 5)

        # Set the sizer for the dialog
        self.SetSizer(dialog_sizer)

    def get_data(self):
        """Get the data entered by the user."""
        return (
            self.name_ctrl.GetValue(),
            self.phone_ctrl.GetValue(),
            self.email_ctrl.GetValue(),
            self.address_ctrl.GetValue()
        )


# Run the application
if __name__ == "__main__":
    app = wx.App(False)
    frame = PhoneBookApp(None, "Phone Book App")
    frame.Show()
    app.MainLoop()
