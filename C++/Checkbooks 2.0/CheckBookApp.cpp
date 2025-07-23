#pragma comment(lib, "C:/Users/mesho/vcpkg/installed/x64-windows/lib/sqlite3.lib")
#include "CheckbookApp.h"
#include <wx/msgdlg.h>

CheckbookApp::CheckbookApp()
    : wxFrame(nullptr, wxID_ANY, "Checkbook Manager", wxDefaultPosition, wxSize(800, 600)),
    db(nullptr) {
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
    ClearTransactionGrid();

    const char* sql = "SELECT date, description, amount, type FROM transactions WHERE person_id = ? ORDER BY date DESC";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, personId);

        double balance = CalculateBalance(personId);
        int row = 0;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            transactionGrid->InsertRows(row);

            const char* date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* desc = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            double amount = sqlite3_column_double(stmt, 2);
            const char* type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));

            transactionGrid->SetCellValue(row, 0, date);
            transactionGrid->SetCellValue(row, 1, desc);

            if (strcmp(type, "credit") == 0) {
                transactionGrid->SetCellValue(row, 2, wxString::Format("$%.2f", amount));
                balance -= amount;
            }
            else {
                transactionGrid->SetCellValue(row, 3, wxString::Format("$%.2f", amount));
                balance += amount;
            }

            transactionGrid->SetCellValue(row, 4, wxString::Format("$%.2f", balance));
            row++;
        }
        sqlite3_finalize(stmt);
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
    if (transactionGrid->GetNumberRows() > 0) {
        transactionGrid->DeleteRows(0, transactionGrid->GetNumberRows());
    }
}

void CheckbookApp::CreateControls() {
    wxPanel* panel = new wxPanel(this);
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

    // Person selection
    wxBoxSizer* personSizer = new wxBoxSizer(wxHORIZONTAL);
    personSizer->Add(new wxStaticText(panel, wxID_ANY, "Select Person:"), 0, wxALIGN_CENTER | wxALL, 5);

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
    transactionGrid->CreateGrid(0, 5);
    transactionGrid->SetColLabelValue(0, "Date");
    transactionGrid->SetColLabelValue(1, "Description");
    transactionGrid->SetColLabelValue(2, "Credit");
    transactionGrid->SetColLabelValue(3, "Debit");
    transactionGrid->SetColLabelValue(4, "Balance");
    mainSizer->Add(transactionGrid, 1, wxEXPAND | wxALL, 5);

    // Transaction form
    wxFlexGridSizer* formSizer = new wxFlexGridSizer(3, 5, 5);
    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Date:"), 0, wxALIGN_CENTER_VERTICAL);
    datePicker = new wxCalendarCtrl(panel, wxID_ANY);
    formSizer->Add(datePicker, 0, wxEXPAND);

    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Description:"), 0, wxALIGN_CENTER_VERTICAL);
    descriptionTxt = new wxTextCtrl(panel, wxID_ANY);
    formSizer->Add(descriptionTxt, 0, wxEXPAND);

    formSizer->Add(new wxStaticText(panel, wxID_ANY, "Amount:"), 0, wxALIGN_CENTER_VERTICAL);
    amountTxt = new wxTextCtrl(panel, wxID_ANY);
    formSizer->Add(amountTxt, 0, wxEXPAND);

    transactionType = new wxChoice(panel, wxID_ANY);
    transactionType->Append("credit");
    transactionType->Append("debit");
    transactionType->SetSelection(0);
    formSizer->Add(transactionType, 0, wxEXPAND);

    addTransactionBtn = new wxButton(panel, wxID_ANY, "Add Transaction");
    formSizer->Add(addTransactionBtn, 0, wxEXPAND);
    mainSizer->Add(formSizer, 0, wxEXPAND | wxALL, 5);

    panel->SetSizer(mainSizer);
}
void CheckbookApp::SetupEventHandlers() {
    // Using dynamic binding instead of event table
    addPersonBtn->Bind(wxEVT_BUTTON, &CheckbookApp::OnAddPerson, this);
    addTransactionBtn->Bind(wxEVT_BUTTON, &CheckbookApp::OnAddTransaction, this);
    personChoice->Bind(wxEVT_CHOICE, &CheckbookApp::OnPersonSelected, this);

    // Note: transactionType is NOT bound to any handler
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
        int personId = wxAtoi(personIdStr);
        LoadTransactions(personId);
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