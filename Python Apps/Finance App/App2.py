import wx
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt

class FinanceTracker(wx.Frame):
    def __init__(self, *args, **kw):
        super(FinanceTracker, self).__init__(*args, **kw)

        self.income_list = []
        self.expense_list = []

        self.InitUI()
        self.LoadData()  # Load data from XML if it exists

    def InitUI(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Income Section
        income_box = wx.BoxSizer(wx.HORIZONTAL)
        self.income_input = wx.TextCtrl(panel)
        self.add_income_btn = wx.Button(panel, label='Add Income')
        self.add_income_btn.Bind(wx.EVT_BUTTON, self.OnAddIncome)
        income_box.Add(self.income_input, proportion=1)
        income_box.Add(self.add_income_btn, flag=wx.LEFT, border=5)
        vbox.Add(income_box, flag=wx.EXPAND | wx.ALL, border=10)

        # Expense Section
        expense_box = wx.BoxSizer(wx.HORIZONTAL)
        self.expense_input = wx.TextCtrl(panel)
        self.add_expense_btn = wx.Button(panel, label='Add Expense')
        self.add_expense_btn.Bind(wx.EVT_BUTTON, self.OnAddExpense)
        expense_box.Add(self.expense_input, proportion=1)
        expense_box.Add(self.add_expense_btn, flag=wx.LEFT, border=5)
        vbox.Add(expense_box, flag=wx.EXPAND | wx.ALL, border=10)

        # Display Section
        self.display_area = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        vbox.Add(self.display_area, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Save/Load Buttons
        save_btn = wx.Button(panel, label='Save to XML')
        load_btn = wx.Button(panel, label='Load from XML')
        save_btn.Bind(wx.EVT_BUTTON, self.OnSave)
        load_btn.Bind(wx.EVT_BUTTON, self.OnLoad)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(save_btn, flag=wx.RIGHT, border=5)
        hbox.Add(load_btn)

        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        # Chart Button
        chart_btn = wx.Button(panel, label='Show Chart')
        chart_btn.Bind(wx.EVT_BUTTON, self.OnShowChart)
        vbox.Add(chart_btn, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        panel.SetSizer(vbox)

        self.SetTitle('Personal Finance Tracker')
        self.SetSize((400, 400))
        self.Centre()

    def OnAddIncome(self, event):
        income = self.income_input.GetValue()
        if income:
            self.income_list.append(float(income))
            self.income_input.Clear()
            self.UpdateDisplay()

    def OnAddExpense(self, event):
        expense = self.expense_input.GetValue()
        if expense:
            self.expense_list.append(float(expense))
            self.expense_input.Clear()
            self.UpdateDisplay()

    def UpdateDisplay(self):
        self.display_area.Clear()
        self.display_area.AppendText("Income:\n")
        for income in self.income_list:
            self.display_area.AppendText(f"{income}\n")
        self.display_area.AppendText("\nExpenses:\n")
        for expense in self.expense_list:
            self.display_area.AppendText(f"{expense}\n")

    def OnSave(self, event):
        root = ET.Element("FinancialData")
        income_elem = ET.SubElement(root, "Income")
        for income in self.income_list:
            ET.SubElement(income_elem, "Item").text = str(income)

        expense_elem = ET.SubElement(root, "Expenses")
        for expense in self.expense_list:
            ET.SubElement(expense_elem, "Item").text = str(expense)

        tree = ET.ElementTree(root)
        with open("financial_data.xml", "wb") as fh:
            tree.write(fh)

    def OnLoad(self, event):
        if os.path.exists("financial_data.xml"):
            tree = ET.parse("financial_data.xml")
            root = tree.getroot()

            self.income_list.clear()
            self.expense_list.clear()

            for income in root.find("Income").findall("Item"):
                self.income_list.append(float(income.text))

            for expense in root.find("Expenses").findall("Item"):
                self.expense_list.append(float(expense.text))

            self.UpdateDisplay()

    def OnShowChart(self, event):
        plt.figure(figsize=(10, 5))
        plt.bar(['Income', 'Expenses'], [sum(self.income_list), sum(self.expense_list)], color=['green', 'red'])
        plt.title('Income vs Expenses')
        plt.ylabel('Amount')
        plt.show()

    def LoadData(self):
        if os.path.exists("financial_data.xml"):
            tree = ET.parse("financial_data.xml")
            root = tree.getroot()

            self.income_list = [item.text for item in root.find("Income").findall("Item")]
            self.expense_list = [item.text for item in root.find("Expenses").findall("Item")]

            self.UpdateDisplay()    

def main():
    app = wx.App()
    tracker = FinanceTracker(None)
    tracker.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()