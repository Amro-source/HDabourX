#pragma once
#include <wx/wx.h>
#include <wx/grid.h>
#include <wx/calctrl.h>
#include <vector>
#include <map>
#include <sqlite3.h>

class CheckbookApp : public wxFrame {
public:
    CheckbookApp();
    virtual ~CheckbookApp();

protected:
    void OnAddPerson(wxCommandEvent& event);
    void OnAddTransaction(wxCommandEvent& event);
    void OnPersonSelected(wxCommandEvent& event);

private:
    // UI Controls
    wxChoice* personChoice;
    wxButton* addPersonBtn;
    wxStaticText* balanceLabel;
    wxGrid* transactionGrid;
    wxCalendarCtrl* datePicker;
    wxTextCtrl* descriptionTxt;
    wxTextCtrl* amountTxt;
    wxChoice* transactionType;
    wxButton* addTransactionBtn;

    // Database
    sqlite3* db;
    void InitDB();
    void CreateTables();
    bool ExecuteSQL(const char* sql);
    void LoadPersons();
    void LoadTransactions(int personId);
    double CalculateBalance(int personId);

    // Methods
    void CreateControls();
    void SetupEventHandlers();
    void UpdateBalanceDisplay();
    void ClearTransactionGrid();
};
