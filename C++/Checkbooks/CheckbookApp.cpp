#include "CheckbookApp.h"



CheckbookApp::CheckbookApp()
    : wxFrame(nullptr, wxID_ANY, "Checkbook Manager", wxDefaultPosition, wxSize(800, 600)) {
    CreateControls();
    SetupEventHandlers();
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


void CheckbookApp::OnAddPerson(wxCommandEvent& event) {
    wxTextEntryDialog dialog(this, "Enter person's name:", "Add Person");
    if (dialog.ShowModal() == wxID_OK) {
        wxString name = dialog.GetValue();
        if (!name.empty()) {
            persons.push_back(name);
            balances[name] = 0.0;
            personChoice->Append(name);
            personChoice->SetSelection(personChoice->GetCount() - 1);
            ClearTransactionGrid();
            UpdateBalanceDisplay();
        }
    }
    event.Skip();
}

void CheckbookApp::OnAddTransaction(wxCommandEvent& event) {
    if (personChoice->GetSelection() == wxNOT_FOUND) {
        wxMessageBox("Please select a person first!", "Error", wxOK | wxICON_ERROR);
        return;
    }

    wxString person = personChoice->GetStringSelection();
    double amount;
    if (!amountTxt->GetValue().ToDouble(&amount) || amount <= 0) {
        wxMessageBox("Please enter a valid positive amount", "Error", wxOK | wxICON_ERROR);
        return;
    }

    // Add transaction to grid
    int row = transactionGrid->GetNumberRows();
    transactionGrid->AppendRows(1);

    wxDateTime date = datePicker->GetDate();
    transactionGrid->SetCellValue(row, 0, date.FormatISODate());
    transactionGrid->SetCellValue(row, 1, descriptionTxt->GetValue());

    // Update balance
    if (transactionType->GetStringSelection() == "credit") {
        balances[person] += amount;
        transactionGrid->SetCellValue(row, 2, wxString::Format("$%.2f", amount));
        transactionGrid->SetCellValue(row, 3, "");
    }
    else {
        balances[person] -= amount;
        transactionGrid->SetCellValue(row, 2, "");
        transactionGrid->SetCellValue(row, 3, wxString::Format("$%.2f", amount));
    }

    // Update balance cell
    transactionGrid->SetCellValue(row, 4, wxString::Format("$%.2f", balances[person]));

    // Update balance display and clear form
    UpdateBalanceDisplay();
    descriptionTxt->Clear();
    amountTxt->Clear();

    event.Skip();
}

void CheckbookApp::OnPersonSelected(wxCommandEvent& event) {
    ClearTransactionGrid();
    UpdateBalanceDisplay();
    event.Skip();
}

void CheckbookApp::UpdateBalanceDisplay() {
    if (personChoice->GetSelection() != wxNOT_FOUND) {
        wxString person = personChoice->GetStringSelection();
        balanceLabel->SetLabel(wxString::Format("Balance: $%.2f", balances[person]));
    }
    else {
        balanceLabel->SetLabel("Balance: $0.00");
    }
}