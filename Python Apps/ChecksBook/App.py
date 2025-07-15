import wx
import wx.adv  # For CalendarCtrl
from wx import grid  # Proper import for Grid
import sqlite3
from datetime import datetime

class CheckbookApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Checkbook Management System", size=(800, 600))
        
        # Initialize database
        self.init_db()
        
        # Create UI
        self.create_ui()
        
        # Load initial data
        self.load_persons()
        self.load_transactions()
        
        self.Show()

    def init_db(self):
        """Initialize the SQLite database and tables"""
        self.conn = sqlite3.connect('checkbook.db')
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                balance REAL DEFAULT 0
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                date TEXT NOT NULL,
                description TEXT,
                amount REAL NOT NULL,
                type TEXT CHECK(type IN ('credit', 'debit')),
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
        ''')
        
        self.conn.commit()

    def create_ui(self):
        """Create the main user interface"""
        panel = wx.Panel(self)
        
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Person selection
        person_sizer = wx.BoxSizer(wx.HORIZONTAL)
        person_sizer.Add(wx.StaticText(panel, label="Select Person:"), 0, wx.ALIGN_CENTER|wx.ALL, 5)
        
        self.person_choice = wx.Choice(panel)
        person_sizer.Add(self.person_choice, 1, wx.EXPAND|wx.ALL, 5)
        
        self.add_person_btn = wx.Button(panel, label="Add Person")
        person_sizer.Add(self.add_person_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(person_sizer, 0, wx.EXPAND|wx.ALL, 5)
        
        # Balance display
        self.balance_label = wx.StaticText(panel, label="Current Balance: $0.00")
        main_sizer.Add(self.balance_label, 0, wx.ALL, 5)
        
        # Transaction grid
        self.transaction_grid = grid.Grid(panel)  # Changed here
        self.transaction_grid.CreateGrid(0, 5)
        self.transaction_grid.SetColLabelValue(0, "Date")
        self.transaction_grid.SetColLabelValue(1, "Description")
        self.transaction_grid.SetColLabelValue(2, "Credit")
        self.transaction_grid.SetColLabelValue(3, "Debit")
        self.transaction_grid.SetColLabelValue(4, "Balance")
        self.transaction_grid.SetColSize(0, 100)
        self.transaction_grid.SetColSize(1, 200)
        self.transaction_grid.SetColSize(2, 100)
        self.transaction_grid.SetColSize(3, 100)
        self.transaction_grid.SetColSize(4, 100)
        main_sizer.Add(self.transaction_grid, 1, wx.EXPAND|wx.ALL, 5)
        
        # Transaction form
        form_sizer = wx.FlexGridSizer(cols=3, vgap=5, hgap=5)
        
        form_sizer.Add(wx.StaticText(panel, label="Date:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.date_picker = wx.adv.CalendarCtrl(panel, style=wx.adv.CAL_SHOW_HOLIDAYS)  # Changed here
        form_sizer.Add(self.date_picker, 0, wx.EXPAND)
        
        form_sizer.Add(wx.StaticText(panel, label="Description:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.description_txt = wx.TextCtrl(panel)
        form_sizer.Add(self.description_txt, 0, wx.EXPAND)
        
        form_sizer.Add(wx.StaticText(panel, label="Amount:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.amount_txt = wx.TextCtrl(panel)
        form_sizer.Add(self.amount_txt, 0, wx.EXPAND)
        
        self.transaction_type = wx.Choice(panel, choices=['credit', 'debit'])
        self.transaction_type.SetSelection(0)
        form_sizer.Add(self.transaction_type, 0, wx.EXPAND)
        
        self.add_transaction_btn = wx.Button(panel, label="Add Transaction")
        form_sizer.Add(self.add_transaction_btn, 0, wx.EXPAND)
        
        main_sizer.Add(form_sizer, 0, wx.EXPAND|wx.ALL, 5)
        
        # Set event handlers
        self.person_choice.Bind(wx.EVT_CHOICE, self.on_person_selected)
        self.add_person_btn.Bind(wx.EVT_BUTTON, self.on_add_person)
        self.add_transaction_btn.Bind(wx.EVT_BUTTON, self.on_add_transaction)
        
        panel.SetSizer(main_sizer)

    # ... [rest of your methods remain exactly the same] ...


    def load_persons(self):
        """Load persons from database into the choice control"""
        self.cursor.execute("SELECT id, name FROM persons")
        persons = self.cursor.fetchall()
        
        self.person_choice.Clear()
        self.person_ids = {}
        
        for person_id, name in persons:
            self.person_choice.Append(name)
            self.person_ids[name] = person_id
        
        if persons:
            self.person_choice.SetSelection(0)
            self.current_person_id = persons[0][0]
            self.update_balance_display()

    def load_transactions(self):
        """Load transactions for the selected person"""
        if not hasattr(self, 'current_person_id'):
            return
            
        # Clear existing rows
        if self.transaction_grid.GetNumberRows() > 0:
            self.transaction_grid.DeleteRows(0, self.transaction_grid.GetNumberRows())
        
        self.cursor.execute('''
            SELECT date, description, amount, type 
            FROM transactions 
            WHERE person_id = ? 
            ORDER BY date DESC
        ''', (self.current_person_id,))
        
        transactions = self.cursor.fetchall()
        
        # Calculate running balance
        balance = 0
        self.cursor.execute("SELECT balance FROM persons WHERE id = ?", (self.current_person_id,))
        balance = self.cursor.fetchone()[0]
        
        # Add transactions to grid in reverse order (newest first)
        for i, (date, desc, amount, ttype) in enumerate(reversed(transactions)):
            self.transaction_grid.InsertRows(0)
            
            if ttype == 'credit':
                credit = f"${amount:.2f}"
                debit = ""
            else:
                credit = ""
                debit = f"${amount:.2f}"
            
            self.transaction_grid.SetCellValue(0, 0, date)
            self.transaction_grid.SetCellValue(0, 1, desc)
            self.transaction_grid.SetCellValue(0, 2, credit)
            self.transaction_grid.SetCellValue(0, 3, debit)
            self.transaction_grid.SetCellValue(0, 4, f"${balance:.2f}")
            
            # Adjust balance based on transaction type
            if ttype == 'credit':
                balance -= amount
            else:
                balance += amount

    def update_balance_display(self):
        """Update the balance label for the current person"""
        if hasattr(self, 'current_person_id'):
            self.cursor.execute("SELECT balance FROM persons WHERE id = ?", (self.current_person_id,))
            balance = self.cursor.fetchone()[0]
            self.balance_label.SetLabel(f"Current Balance: ${balance:.2f}")

    def on_person_selected(self, event):
        """Handle person selection change"""
        selected_name = self.person_choice.GetString(self.person_choice.GetSelection())
        self.current_person_id = self.person_ids[selected_name]
        self.load_transactions()
        self.update_balance_display()

    def on_add_person(self, event):
        """Add a new person to the database"""
        dlg = wx.TextEntryDialog(self, "Enter new person's name:", "Add Person")
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetValue()
            if name:
                try:
                    self.cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
                    self.conn.commit()
                    self.load_persons()
                except sqlite3.IntegrityError:
                    wx.MessageBox("Person with this name already exists!", "Error", wx.OK|wx.ICON_ERROR)
        dlg.Destroy()

    def on_add_transaction(self, event):
        """Add a new transaction for the current person"""
        if not hasattr(self, 'current_person_id'):
            wx.MessageBox("Please select a person first!", "Error", wx.OK|wx.ICON_ERROR)
            return
            
        try:
            amount = float(self.amount_txt.GetValue())
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except ValueError:
            wx.MessageBox("Please enter a valid positive amount", "Error", wx.OK|wx.ICON_ERROR)
            return
            
        description = self.description_txt.GetValue()
        if not description:
            wx.MessageBox("Please enter a description", "Error", wx.OK|wx.ICON_ERROR)
            return
            
        date = self.date_picker.GetDate().FormatISODate()
        ttype = self.transaction_type.GetString(self.transaction_type.GetSelection())
        
        # Update balance in persons table
        if ttype == 'credit':
            self.cursor.execute("UPDATE persons SET balance = balance + ? WHERE id = ?", 
                              (amount, self.current_person_id))
        else:
            self.cursor.execute("UPDATE persons SET balance = balance - ? WHERE id = ?", 
                              (amount, self.current_person_id))
        
        # Add transaction
        self.cursor.execute('''
            INSERT INTO transactions (person_id, date, description, amount, type)
            VALUES (?, ?, ?, ?, ?)
        ''', (self.current_person_id, date, description, amount, ttype))
        
        self.conn.commit()
        
        # Clear form and refresh display
        self.amount_txt.Clear()
        self.description_txt.Clear()
        self.load_transactions()
        self.update_balance_display()

    def __del__(self):
        """Clean up when closing"""
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    app = wx.App(False)
    frame = CheckbookApp()
    app.MainLoop()