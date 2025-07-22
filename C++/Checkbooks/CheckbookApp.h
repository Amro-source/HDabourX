#pragma once
#include <wx/wx.h>
#include <wx/grid.h>
#include <wx/calctrl.h>
#include <vector>
#include <map>

class CheckbookApp : public wxFrame {
public:
    CheckbookApp();
    virtual ~CheckbookApp() = default;

protected:  // Changed from private to allow event handler access
    // Event handlers - must be protected or public for wxWidgets
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

    // Data structures
    std::vector<wxString> persons;
    std::map<wxString, double> balances;

    // Methods
    void CreateControls();
    void SetupEventHandlers();
    void UpdateBalanceDisplay();
    void ClearTransactionGrid();

    // No event table macros needed
};