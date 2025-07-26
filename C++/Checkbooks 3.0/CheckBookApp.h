#pragma once
#include <wx/wx.h>
#include <wx/grid.h>
#include <wx/calctrl.h>
#include <wx/renderer.h>  // Add this for wxRendererNative
#include <vector>
#include <map>
#include <sqlite3.h>

class GridButtonRenderer : public wxGridCellRenderer {
public:
    GridButtonRenderer(const wxString& label) : m_label(label) {}

    void Draw(wxGrid& grid, wxGridCellAttr& attr, wxDC& dc,
        const wxRect& rect, int row, int col, bool isSelected) override {
        wxRect butRect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height - 4);
        wxRendererNative::GetDefault().DrawPushButton(
            &grid, dc, butRect, 0);
        dc.DrawLabel(m_label, butRect, wxALIGN_CENTER);
    }

    wxSize GetBestSize(wxGrid& grid, wxGridCellAttr& attr, wxDC& dc,
        int row, int col) override {
        return wxSize(60, 25);
    }

    wxGridCellRenderer* Clone() const override {
        return new GridButtonRenderer(*this);
    }

private:
    wxString m_label;
};

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
    int m_currentPersonId;  // Store currently selected person ID

    // Methods
    wxWindow* CreateGridButton(wxWindow* parent, const wxString& label, int row);
    void UpdateGridButtons();

    // Database
    sqlite3* db;
    void InitDB();
    void CreateTables();
    bool ExecuteSQL(const char* sql);
    void LoadPersons();
    void LoadTransactions(int personId);
    double CalculateBalance(int personId);

    void CreateControls();
    void SetupEventHandlers();
    void UpdateBalanceDisplay();
    void ClearTransactionGrid();

    // Event handlers
    void OnEditTransaction(int row);
    void OnDeleteTransaction(int row);
    void OnGridCellClick(wxGridEvent& event);
    void ShowContextMenu(int row, const wxPoint& pos);
    void ShowEditDialog(int row);
};