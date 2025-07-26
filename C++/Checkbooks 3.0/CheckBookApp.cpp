#pragma comment(lib, "C:/Users/mesho/vcpkg/installed/x64-windows/lib/sqlite3.lib")
#include "CheckbookApp.h"
#include <wx/msgdlg.h>
#include <wx/grid.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/dialog.h>
#include <wx/msgdlg.h>
#include <sqlite3.h>
CheckbookApp::CheckbookApp() : wxFrame(nullptr, wxID_ANY, "Checkbook Manager"),
m_currentPersonId(-1) {
    InitDB();
    CreateControls();
    SetupEventHandlers();
    LoadPersons();
}

CheckbookApp::~CheckbookApp() {
    if (db) sqlite3_close(db);
}

// Database Initialization
void CheckbookApp::InitDB() {
    if (sqlite3_open("checkbook.db", &db) != SQLITE_OK) {
        wxMessageBox("Cannot open database", "Error", wxOK | wxICON_ERROR);
        return;
    }
    CreateTables();
}

void CheckbookApp::CreateTables() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            description TEXT,
            amount REAL NOT NULL,
            type TEXT CHECK(type IN ('credit', 'debit')),
            FOREIGN KEY(person_id) REFERENCES persons(id)
        );
    )";
    ExecuteSQL(sql);
}

bool CheckbookApp::ExecuteSQL(const char* sql) {
    char* errMsg = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        wxMessageBox(errMsg, "SQL Error", wxOK | wxICON_ERROR);
        sqlite3_free(errMsg);
        return false;
    }
    return true;
}

// Data Loading
void CheckbookApp::LoadPersons() {
    personChoice->Clear();

    const char* sql = "SELECT id, name FROM persons ORDER BY name";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);
            const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            personChoice->Append(name, new wxStringClientData(wxString::Format("%d", id)));
        }
        sqlite3_finalize(stmt);
    }

    if (personChoice->GetCount() > 0) {
        personChoice->SetSelection(0);
        wxCommandEvent evt;
        OnPersonSelected(evt);
    }
}

void CheckbookApp::LoadTransactions(int personId) {
    // Clear existing grid data
    ClearTransactionGrid();

   // wxString idMsg = "LoadTransactions called for person ID: " + wxString::Format("%d", personId);
   // wxMessageBox(idMsg);

    const char* sql = "SELECT id, date, description, amount, type FROM transactions WHERE person_id = ? ORDER BY date ASC";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, personId);
        //wxMessageBox("SQL query prepared and bound correctly");

        double runningBalance = 0.0;
        int row = 0;
        int count = 0;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            ++count;

            transactionGrid->InsertRows(row);

            int transId = sqlite3_column_int(stmt, 0);
            const char* date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* desc = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            double amount = sqlite3_column_double(stmt, 3);
            const char* type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));

            // Column 0 = ID (hidden)
            transactionGrid->SetCellValue(row, 0, wxString::Format("%d", transId));
            transactionGrid->SetCellValue(row, 1, date); // Date
            transactionGrid->SetCellValue(row, 2, desc); // Description

            if (strcmp(type, "credit") == 0) {
                runningBalance += amount;
                transactionGrid->SetCellValue(row, 3, wxString::Format("$%.2f", amount)); // Credit
                transactionGrid->SetCellValue(row, 4, ""); // Debit
            }
            else {
                runningBalance -= amount;
                transactionGrid->SetCellValue(row, 3, ""); // Credit
                transactionGrid->SetCellValue(row, 4, wxString::Format("$%.2f", amount)); // Debit
            }

            transactionGrid->SetCellValue(row, 5, wxString::Format("$%.2f", runningBalance)); // Balance

            // Add dynamic buttons
            transactionGrid->SetCellRenderer(row, 6, new GridButtonRenderer("Edit"));
            transactionGrid->SetCellRenderer(row, 7, new GridButtonRenderer("Delete"));

            row++;
        }

        sqlite3_finalize(stmt);

       // wxMessageBox("Transactions loaded: " + wxString::Format("%d", count));
    }
    else {
        wxMessageBox("SQL prepare failed!");
    }
}

// Add this with your other event handlers
void CheckbookApp::OnGridCellClick(wxGridEvent& event) {
    int row = event.GetRow();
    int col = event.GetCol();

    if (col == 6) {  // Edit button column
        ShowEditDialog(row);
    }
    else if (col == 7) {  // Delete button column
        OnDeleteTransaction(row);
    }

    event.Skip(); // Let other handlers run
}

// Add this with your other methods
void CheckbookApp::ShowContextMenu(int row, const wxPoint& pos) {
    wxMenu menu;
    menu.Append(1, "Edit Transaction");
    menu.Append(2, "Delete Transaction");

    menu.Bind(wxEVT_MENU, [this, row](wxCommandEvent& event) {
        wxString transId = transactionGrid->GetCellValue(row, 0);

        if (event.GetId() == 1) {
            // Edit transaction - implement your edit dialog here
            wxMessageBox("Editing transaction ID: " + transId, "Edit");
        }
        else {
            // Delete transaction
            if (wxMessageBox("Delete transaction " + transId + "?", "Confirm",
                wxYES_NO | wxICON_QUESTION) == wxYES) {
                // Execute DELETE SQL here
                const char* sql = "DELETE FROM transactions WHERE id=?";
                sqlite3_stmt* stmt;
                if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
                    sqlite3_bind_text(stmt, 1, transId.ToUTF8(), -1, SQLITE_TRANSIENT);
                    if (sqlite3_step(stmt) == SQLITE_DONE) {
                        LoadTransactions(m_currentPersonId); // Refresh grid
                    }
                    sqlite3_finalize(stmt);
                }
            }
        }
        });

    transactionGrid->PopupMenu(&menu, pos);
}
void CheckbookApp::UpdateGridButtons() {
    int rowCount = transactionGrid->GetNumberRows();
    for (int row = 0; row < rowCount; ++row) {
        // Ensure these are set only on columns 6 and 7 (Edit/Delete)
        transactionGrid->SetCellRenderer(row, 6, new GridButtonRenderer("Edit"));
        transactionGrid->SetCellRenderer(row, 7, new GridButtonRenderer("Delete"));
    }
}


// New event handlers (modified to take row parameter)
//void CheckbookApp::OnEditTransaction(int row) {
   // if (row < 0 || row >= transactionGrid->GetNumberRows()) return;

  //  wxString transactionId = transactionGrid->GetCellValue(row, 0); // ID column

   // wxDialog dlg(this, wxID_ANY, "Edit Transaction", wxDefaultPosition, wxSize(400, 300));
  //  wxPanel* panel = new wxPanel(&dlg);

    // Add your edit controls here...

  //  if (dlg.ShowModal() == wxID_OK) {
        // Update database
       // const char* sql = "UPDATE transactions SET description=? WHERE id=?";
       // sqlite3_stmt* stmt;
      // if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
         //   sqlite3_bind_text(stmt, 1, "Updated Description", -1, SQLITE_TRANSIENT); // Replace with actual value
         //   sqlite3_bind_text(stmt, 2, transactionId.ToUTF8(), -1, SQLITE_TRANSIENT);

           // if (sqlite3_step(stmt) == SQLITE_DONE) {
          //      LoadTransactions(m_currentPersonId);
       //     }
        //    sqlite3_finalize(stmt);
       // }
    //}
//}

void CheckbookApp::ShowEditDialog(int row) {
    if (row < 0 || row >= transactionGrid->GetNumberRows()) return;

    // Get transaction data from grid
    wxString transId = transactionGrid->GetCellValue(row, 0);
    wxString dateStr = transactionGrid->GetCellValue(row, 1);
    wxString description = transactionGrid->GetCellValue(row, 2);
    wxString amountStr;
    wxString type;

    // Determine if this is credit or debit
    if (!transactionGrid->GetCellValue(row, 3).IsEmpty()) {
        amountStr = transactionGrid->GetCellValue(row, 3).AfterFirst('$');
        type = "credit";
    }
    else {
        amountStr = transactionGrid->GetCellValue(row, 4).AfterFirst('$');
        type = "debit";
    }

    // Create dialog
    wxDialog dlg(this, wxID_ANY, "Edit Transaction", wxDefaultPosition, wxSize(400, 300));
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    wxPanel* panel = new wxPanel(&dlg);

    // Create controls
    wxFlexGridSizer* formSizer = new wxFlexGridSizer(2, 5, 5);

    // Date control
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Date:"), 0, wxALIGN_CENTER_VERTICAL);
    wxCalendarCtrl* datePicker = new wxCalendarCtrl(panel, wxID_ANY);
    wxDateTime date;
    if (date.ParseISODate(dateStr)) {
        datePicker->SetDate(date);
    }
    formSizer->Add(datePicker, 0, wxEXPAND);

    // Description control
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Description:"), 0, wxALIGN_CENTER_VERTICAL);
    wxTextCtrl* descCtrl = new wxTextCtrl(panel, wxID_ANY, description);
    formSizer->Add(descCtrl, 0, wxEXPAND);

    // Amount control
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Amount:"), 0, wxALIGN_CENTER_VERTICAL);
    wxTextCtrl* amountCtrl = new wxTextCtrl(panel, wxID_ANY, amountStr);
    formSizer->Add(amountCtrl, 0, wxEXPAND);

    // Type control
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Type:"), 0, wxALIGN_CENTER_VERTICAL);
    wxChoice* typeChoice = new wxChoice(panel, wxID_ANY);
    typeChoice->Append("credit");
    typeChoice->Append("debit");
    typeChoice->SetStringSelection(type);
    formSizer->Add(typeChoice, 0, wxEXPAND);

    // Buttons
    wxBoxSizer* btnSizer = new wxBoxSizer(wxHORIZONTAL);
    wxButton* saveBtn = new wxButton(panel, wxID_OK, "Save");
    wxButton* cancelBtn = new wxButton(panel, wxID_CANCEL, "Cancel");
    btnSizer->Add(saveBtn, 0, wxALL, 5);
    btnSizer->Add(cancelBtn, 0, wxALL, 5);

    // Layout
    panel->SetSizer(new wxBoxSizer(wxVERTICAL));
    panel->GetSizer()->Add(formSizer, 1, wxEXPAND | wxALL, 10);
    panel->GetSizer()->Add(btnSizer, 0, wxALIGN_CENTER | wxBOTTOM, 10);

    dlg.SetSizer(mainSizer);
    mainSizer->Add(panel, 1, wxEXPAND);
    dlg.Layout();
    dlg.Fit();

    if (dlg.ShowModal() == wxID_OK) {
        // Get updated values
        wxString newDate = datePicker->GetDate().FormatISODate();
        wxString newDesc = descCtrl->GetValue();
        wxString newAmountStr = amountCtrl->GetValue();
        wxString newType = typeChoice->GetStringSelection();

        double newAmount;
        if (!newAmountStr.ToDouble(&newAmount) ){
            wxMessageBox("Please enter a valid amount", "Error", wxOK | wxICON_ERROR);
            return;
        }

        // Update database
        const char* sql = R"(
            UPDATE transactions 
            SET date = ?, description = ?, amount = ?, type = ?
            WHERE id = ?
        )";

            sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, newDate.ToUTF8(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, newDesc.ToUTF8(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_double(stmt, 3, newAmount);
            sqlite3_bind_text(stmt, 4, newType.ToUTF8(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 5, transId.ToUTF8(), -1, SQLITE_TRANSIENT);

            if (sqlite3_step(stmt) == SQLITE_DONE) {
                // Refresh the grid
                if (m_currentPersonId != -1) {
                    LoadTransactions(m_currentPersonId);
                    UpdateBalanceDisplay();
                }
            }
            else {
                wxMessageBox("Failed to update transaction", "Error", wxOK | wxICON_ERROR);
            }
            sqlite3_finalize(stmt);
        }
    }
}

void CheckbookApp::OnEditTransaction(int row) {
    ShowEditDialog(row);
}

void CheckbookApp::OnDeleteTransaction(int row) {
    if (row < 0 || row >= transactionGrid->GetNumberRows()) return;

    // Get the transaction ID from hidden column 0
    wxString transactionId = transactionGrid->GetCellValue(row, 0);
    if (transactionId.IsEmpty()) {
        wxMessageBox("Error: Transaction ID is empty.");
        return;
    }

    if (wxMessageBox("Delete this transaction?", "Confirm", wxYES_NO | wxICON_WARNING) == wxYES) {
        const char* sql = "DELETE FROM transactions WHERE id = ?";
        sqlite3_stmt* stmt;

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, transactionId.ToUTF8(), -1, SQLITE_TRANSIENT);

            if (sqlite3_step(stmt) == SQLITE_DONE) {
                // Success - reload transactions for current person
                if (m_currentPersonId != -1) {
                    LoadTransactions(m_currentPersonId);
                    UpdateBalanceDisplay();
                }
            }
            else {
                wxMessageBox("Error: Could not delete transaction.");
            }
            sqlite3_finalize(stmt);
        }
        else {
            wxMessageBox("Error: Failed to prepare DELETE statement.");
        }
    }
}




double CheckbookApp::CalculateBalance(int personId) {
    const char* sql = R"(
        SELECT SUM(CASE WHEN type='credit' THEN amount ELSE -amount END) 
        FROM transactions WHERE person_id = ?
    )";
    sqlite3_stmt* stmt;
    double balance = 0.0;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, personId);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            balance = sqlite3_column_double(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    return balance;
}

void CheckbookApp::ClearTransactionGrid() {
    int rows = transactionGrid->GetNumberRows();
    if (rows > 0) transactionGrid->DeleteRows(0, rows);
}

void CheckbookApp::CreateControls() {
    wxPanel* panel = new wxPanel(this);
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

    // Person selection section
    wxBoxSizer* personSizer = new wxBoxSizer(wxHORIZONTAL);
    personSizer->Add(new wxStaticText(panel, wxID_ANY, "Select Person:"),
        0, wxALIGN_CENTER_VERTICAL | wxALL, 5);

    personChoice = new wxChoice(panel, wxID_ANY);
    personSizer->Add(personChoice, 1, wxEXPAND | wxALL, 5);

    addPersonBtn = new wxButton(panel, wxID_ANY, "Add Person");
    personSizer->Add(addPersonBtn, 0, wxALL, 5);
    mainSizer->Add(personSizer, 0, wxEXPAND | wxALL, 5);

    // Balance display
    balanceLabel = new wxStaticText(panel, wxID_ANY, "Balance: $0.00");
    mainSizer->Add(balanceLabel, 0, wxALL, 5);

    // Transaction grid
    transactionGrid = new wxGrid(panel, wxID_ANY);
    transactionGrid->CreateGrid(0, 8);  // Including hidden ID column
    transactionGrid->HideCol(0); // Hide ID column

    // Configure grid columns
    const wxString colLabels[] = { "ID", "Date", "Description", "Credit", "Debit", "Balance", "Edit", "Delete" };
    for (int i = 0; i < 8; ++i) {
        transactionGrid->SetColLabelValue(i, colLabels[i]);
    }

    // Set column sizes
    transactionGrid->SetColSize(1, 100); // Date
    transactionGrid->SetColSize(2, 200); // Description
    transactionGrid->SetColSize(3, 80);  // Credit
    transactionGrid->SetColSize(4, 80);  // Debit
    transactionGrid->SetColSize(5, 80);  // Balance
    transactionGrid->SetColSize(6, 60);  // Edit
    transactionGrid->SetColSize(7, 60);  // Delete

    transactionGrid->EnableGridLines(false);
    transactionGrid->SetCellHighlightPenWidth(0);
    transactionGrid->SetDefaultCellAlignment(wxALIGN_CENTER, wxALIGN_CENTER);
    mainSizer->Add(transactionGrid, 1, wxEXPAND | wxALL, 5);

    // Transaction form - PROPERLY ALIGNED VERSION
    wxStaticBoxSizer* formBox = new wxStaticBoxSizer(wxVERTICAL, panel, "Add New Transaction");
    wxFlexGridSizer* formSizer = new wxFlexGridSizer(2, 5, 5); // 2 cols, 5px gaps
    formSizer->AddGrowableCol(1, 1); // Make second column expandable

    // Date row
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Date:"),
        0, wxALIGN_CENTER_VERTICAL | wxALIGN_RIGHT | wxRIGHT, 5);
    datePicker = new wxCalendarCtrl(panel, wxID_ANY);
    formSizer->Add(datePicker, 0, wxEXPAND);

    // Description row - perfectly aligned
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Description:"),
        0, wxALIGN_CENTER_VERTICAL | wxALIGN_RIGHT | wxRIGHT, 5);
    descriptionTxt = new wxTextCtrl(panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize);
    formSizer->Add(descriptionTxt, 1, wxEXPAND);

    // Amount row - perfectly aligned
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Amount:"),
        0, wxALIGN_CENTER_VERTICAL | wxALIGN_RIGHT | wxRIGHT, 5);

    wxBoxSizer* amountTypeSizer = new wxBoxSizer(wxHORIZONTAL);
    amountTxt = new wxTextCtrl(panel, wxID_ANY, "", wxDefaultPosition, wxSize(80, -1));
    amountTypeSizer->Add(amountTxt, 0, wxRIGHT, 10);

    transactionType = new wxChoice(panel, wxID_ANY);
    transactionType->Append("credit");
    transactionType->Append("debit");
    transactionType->SetSelection(0);
    amountTypeSizer->Add(transactionType, 0, wxEXPAND);

    formSizer->Add(amountTypeSizer, 0, wxEXPAND);

    // Add Transaction button
    addTransactionBtn = new wxButton(panel, wxID_ANY, "Add Transaction");

    // Add to layouts
    formBox->Add(formSizer, 0, wxEXPAND | wxALL, 10);
    formBox->Add(addTransactionBtn, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);

    // Add to main sizer
    mainSizer->Add(formBox, 0, wxEXPAND | wxALL, 5);

    // Final panel setup
    panel->SetSizer(mainSizer);
    this->SetMinSize(wxSize(600, 500));

    // Set default date and focus
    datePicker->SetDate(wxDateTime::Today());
    descriptionTxt->SetFocus();
}
void CheckbookApp::SetupEventHandlers() {
    // Bind button events
    addPersonBtn->Bind(wxEVT_BUTTON, &CheckbookApp::OnAddPerson, this);
    addTransactionBtn->Bind(wxEVT_BUTTON, &CheckbookApp::OnAddTransaction, this);
    personChoice->Bind(wxEVT_CHOICE, &CheckbookApp::OnPersonSelected, this);

    // Single handler for grid cell clicks (Edit/Delete buttons)
    transactionGrid->Bind(wxEVT_GRID_CELL_LEFT_CLICK, &CheckbookApp::OnGridCellClick, this);

    // Handle right-click for context menu
    transactionGrid->Bind(wxEVT_GRID_CELL_RIGHT_CLICK, [this](wxGridEvent& event) {
        ShowContextMenu(event.GetRow(), event.GetPosition());
        event.Skip();
        });

    // Update grid buttons when rows are resized
    transactionGrid->Bind(wxEVT_GRID_ROW_SIZE, [this](wxGridSizeEvent& event) {
        UpdateGridButtons();
        event.Skip();
        });

    // Optional: Handle row insertions/deletions if needed
    /*
    transactionGrid->Bind(wxEVT_GRID_ROW_INSERTED, [this](wxGridEvent& event) {
        UpdateGridButtons();
        event.Skip();
    });
    transactionGrid->Bind(wxEVT_GRID_ROW_DELETED, [this](wxGridEvent& event) {
        UpdateGridButtons();
        event.Skip();
    });
    */
}


// Modified CreateGridButton
wxWindow* CheckbookApp::CreateGridButton(wxWindow* parent, const wxString& label, int row) {
    wxButton* button = new wxButton(parent, wxID_ANY, label, wxDefaultPosition, wxSize(60, 25));

    // Make a local copy of row & label
    const int capturedRow = row;
    const wxString capturedLabel = label;

    button->Bind(wxEVT_BUTTON, [=](wxCommandEvent&) {
        if (capturedLabel == "Edit") {
            OnEditTransaction(capturedRow);
        }
        else if (capturedLabel == "Delete") {
            OnDeleteTransaction(capturedRow);
        }
        });

    return button;
}

// Event Handlers
void CheckbookApp::OnAddPerson(wxCommandEvent& event) {
    wxTextEntryDialog dialog(this, "Enter person's name:", "Add Person");
    if (dialog.ShowModal() == wxID_OK) {
        wxString name = dialog.GetValue();
        if (!name.empty()) {
            const char* sql = "INSERT INTO persons (name) VALUES (?)";
            sqlite3_stmt* stmt;

            if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
                sqlite3_bind_text(stmt, 1, name.ToUTF8(), -1, SQLITE_TRANSIENT);

                if (sqlite3_step(stmt) == SQLITE_DONE) {
                    LoadPersons(); // Refresh the list
                }
                else {
                    wxMessageBox("Failed to add person", "Error", wxOK | wxICON_ERROR);
                }
                sqlite3_finalize(stmt);
            }
        }
    }
}

void CheckbookApp::OnAddTransaction(wxCommandEvent& event) {
    if (personChoice->GetSelection() == wxNOT_FOUND) {
        wxMessageBox("Please select a person first!", "Error", wxOK | wxICON_ERROR);
        return;
    }

    wxString personIdStr = dynamic_cast<wxStringClientData*>(
        personChoice->GetClientObject(personChoice->GetSelection()))->GetData();
    int personId = wxAtoi(personIdStr);

    wxString description = descriptionTxt->GetValue();
    double amount;
    if (!amountTxt->GetValue().ToDouble(&amount) || amount <= 0) {
        wxMessageBox("Please enter a valid positive amount", "Error", wxOK | wxICON_ERROR);
        return;
    }

    wxString date = datePicker->GetDate().FormatISODate();
    wxString type = transactionType->GetStringSelection();

    const char* sql = R"(
        INSERT INTO transactions (person_id, date, description, amount, type)
        VALUES (?, ?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, personId);
        sqlite3_bind_text(stmt, 2, date.ToUTF8(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, description.ToUTF8(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 4, amount);
        sqlite3_bind_text(stmt, 5, type.ToUTF8(), -1, SQLITE_TRANSIENT);

        if (sqlite3_step(stmt) == SQLITE_DONE) {
            LoadTransactions(personId); // Refresh transactions
            UpdateBalanceDisplay();

            // Clear form
            descriptionTxt->Clear();
            amountTxt->Clear();
        }
        else {
            wxMessageBox("Failed to add transaction", "Error", wxOK | wxICON_ERROR);
        }
        sqlite3_finalize(stmt);
    }
}


void CheckbookApp::OnPersonSelected(wxCommandEvent& event) {
    if (personChoice->GetSelection() != wxNOT_FOUND) {
        wxString personIdStr = dynamic_cast<wxStringClientData*>(
            personChoice->GetClientObject(personChoice->GetSelection()))->GetData();
        m_currentPersonId = wxAtoi(personIdStr);
        LoadTransactions(m_currentPersonId);
        UpdateBalanceDisplay();
    }
}

void CheckbookApp::UpdateBalanceDisplay() {
    if (personChoice->GetSelection() != wxNOT_FOUND) {
        wxString personIdStr = dynamic_cast<wxStringClientData*>(
            personChoice->GetClientObject(personChoice->GetSelection()))->GetData();
        int personId = wxAtoi(personIdStr);
        double balance = CalculateBalance(personId);
        balanceLabel->SetLabel(wxString::Format("Balance: $%.2f", balance));
    }
    else {
        balanceLabel->SetLabel("Balance: $0.00");
    }
}