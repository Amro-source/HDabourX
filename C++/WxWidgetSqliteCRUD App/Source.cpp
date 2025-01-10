#include <wx/wx.h>
#include <wx/grid.h>
#include <sqlite3.h>
#include <pugixml.hpp>

class MyApp : public wxApp {
public:
    virtual bool OnInit();
};

class MyFrame : public wxFrame {
public:
    MyFrame(const wxString& title);

private:
    void OnCreate(wxCommandEvent& event);
    void OnRead(wxCommandEvent& event);
    void OnUpdate(wxCommandEvent& event);
    void OnDelete(wxCommandEvent& event);
    void OnExportXML(wxCommandEvent& event);
    void OnSubmit(wxCommandEvent& event);
    void OnClear(wxCommandEvent& event);

    sqlite3* db;
    wxGrid* grid;
    wxTextCtrl* idCtrl;
    wxTextCtrl* nameCtrl;
    wxTextCtrl* ageCtrl;

    void InitDatabase();
    void LoadDataToGrid();
};

enum ButtonIDs {
    ID_SUBMIT = wxID_HIGHEST + 1,
    ID_CLEAR,
    ID_READ,
    ID_UPDATE,
    ID_DELETE,
    ID_EXPORT
};

IMPLEMENT_APP(MyApp)

bool MyApp::OnInit() {
    MyFrame* frame = new MyFrame("WxWidgets SQLite CRUD Application");
    frame->Show(true);
    return true;
}

MyFrame::MyFrame(const wxString& title) : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(800, 600)) {
    wxPanel* panel = new wxPanel(this, wxID_ANY);

    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

    // Input Fields
    wxBoxSizer* inputSizer = new wxBoxSizer(wxHORIZONTAL);

    inputSizer->Add(new wxStaticText(panel, wxID_ANY, "ID:"), 0, wxALL | wxCENTER, 5);
    idCtrl = new wxTextCtrl(panel, wxID_ANY);
    inputSizer->Add(idCtrl, 1, wxALL | wxEXPAND, 5);

    inputSizer->Add(new wxStaticText(panel, wxID_ANY, "Name:"), 0, wxALL | wxCENTER, 5);
    nameCtrl = new wxTextCtrl(panel, wxID_ANY);
    inputSizer->Add(nameCtrl, 1, wxALL | wxEXPAND, 5);

    inputSizer->Add(new wxStaticText(panel, wxID_ANY, "Age:"), 0, wxALL | wxCENTER, 5);
    ageCtrl = new wxTextCtrl(panel, wxID_ANY);
    inputSizer->Add(ageCtrl, 1, wxALL | wxEXPAND, 5);

    mainSizer->Add(inputSizer, 0, wxEXPAND | wxALL, 10);

    // Buttons for Submission and Clear
    wxBoxSizer* actionSizer = new wxBoxSizer(wxHORIZONTAL);
    actionSizer->Add(new wxButton(panel, ID_SUBMIT, "Submit"), 0, wxALL, 5);
    actionSizer->Add(new wxButton(panel, ID_CLEAR, "Clear"), 0, wxALL, 5);
    mainSizer->Add(actionSizer, 0, wxALIGN_CENTER);

    // Grid
    grid = new wxGrid(panel, wxID_ANY);
    grid->CreateGrid(0, 3);
    grid->SetColLabelValue(0, "ID");
    grid->SetColLabelValue(1, "Name");
    grid->SetColLabelValue(2, "Age");
    mainSizer->Add(grid, 1, wxEXPAND | wxALL, 10);

    // CRUD Buttons
    wxBoxSizer* btnSizer = new wxBoxSizer(wxHORIZONTAL);
    btnSizer->Add(new wxButton(panel, wxID_ADD, "Create"), 0, wxALL, 5);
    btnSizer->Add(new wxButton(panel, ID_READ, "Read"), 0, wxALL, 5);
    btnSizer->Add(new wxButton(panel, ID_UPDATE, "Update"), 0, wxALL, 5);
    btnSizer->Add(new wxButton(panel, ID_DELETE, "Delete"), 0, wxALL, 5);
    btnSizer->Add(new wxButton(panel, ID_EXPORT, "Export to XML"), 0, wxALL, 5);
    mainSizer->Add(btnSizer, 0, wxALIGN_CENTER);

    panel->SetSizer(mainSizer);

    // Bind Events
    Bind(wxEVT_BUTTON, &MyFrame::OnSubmit, this, ID_SUBMIT);
    Bind(wxEVT_BUTTON, &MyFrame::OnClear, this, ID_CLEAR);
    Bind(wxEVT_BUTTON, &MyFrame::OnCreate, this, wxID_ADD);
    Bind(wxEVT_BUTTON, &MyFrame::OnRead, this, ID_READ);
    Bind(wxEVT_BUTTON, &MyFrame::OnUpdate, this, ID_UPDATE);
    Bind(wxEVT_BUTTON, &MyFrame::OnDelete, this, ID_DELETE);
    Bind(wxEVT_BUTTON, &MyFrame::OnExportXML, this, ID_EXPORT);

    InitDatabase();
    LoadDataToGrid();
}

void MyFrame::InitDatabase() {
    if (sqlite3_open("application.db", &db) != SQLITE_OK) {
        wxMessageBox("Failed to open database!", "Error", wxOK | wxICON_ERROR);
        Close(true);
    }

    const char* createTableQuery = "CREATE TABLE IF NOT EXISTS Users (ID INTEGER PRIMARY KEY, Name TEXT, Age INTEGER);";
    char* errMsg = nullptr;
    if (sqlite3_exec(db, createTableQuery, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        wxMessageBox(wxString::Format("Failed to create table: %s", errMsg), "Error", wxOK | wxICON_ERROR);
        sqlite3_free(errMsg);
        Close(true);
    }
}

void MyFrame::LoadDataToGrid() {
    grid->ClearGrid();
    while (grid->GetNumberRows() > 0) {
        grid->DeleteRows(0);
    }

    const char* selectQuery = "SELECT * FROM Users;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, selectQuery, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int row = grid->GetNumberRows();
            grid->AppendRows(1);
            grid->SetCellValue(row, 0, wxString::Format("%d", sqlite3_column_int(stmt, 0)));
            grid->SetCellValue(row, 1, wxString::FromUTF8((const char*)sqlite3_column_text(stmt, 1)));
            grid->SetCellValue(row, 2, wxString::Format("%d", sqlite3_column_int(stmt, 2)));
        }
        sqlite3_finalize(stmt);
    }
}

void MyFrame::OnSubmit(wxCommandEvent& event) {
    wxString id = idCtrl->GetValue();
    wxString name = nameCtrl->GetValue();
    wxString age = ageCtrl->GetValue();

    if (name.IsEmpty() || age.IsEmpty()) {
        wxMessageBox("Please fill in all fields.", "Error", wxOK | wxICON_ERROR);
        return;
    }

    const char* insertQuery = "INSERT INTO Users (ID, Name, Age) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, insertQuery, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, wxAtoi(id));
        sqlite3_bind_text(stmt, 2, name.ToUTF8().data(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 3, wxAtoi(age));
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        LoadDataToGrid();
    }
    else {
        wxMessageBox("Failed to insert data!", "Error", wxOK | wxICON_ERROR);
    }
}

void MyFrame::OnClear(wxCommandEvent& event) {
    idCtrl->Clear();
    nameCtrl->Clear();
    ageCtrl->Clear();
}

void MyFrame::OnCreate(wxCommandEvent& event) {
    LoadDataToGrid();
}

void MyFrame::OnRead(wxCommandEvent& event) {
    LoadDataToGrid();
}

void MyFrame::OnUpdate(wxCommandEvent& event) {
    int row = grid->GetGridCursorRow();
    if (row >= 0) {
        wxString id = grid->GetCellValue(row, 0);
        wxString name = nameCtrl->GetValue();
        wxString age = ageCtrl->GetValue();

        const char* updateQuery = "UPDATE Users SET Name = ?, Age = ? WHERE ID = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, updateQuery, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, name.ToUTF8().data(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(stmt, 2, wxAtoi(age));
            sqlite3_bind_int(stmt, 3, wxAtoi(id));
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
            LoadDataToGrid();
        }
        else {
            wxMessageBox("Failed to update data!", "Error", wxOK | wxICON_ERROR);
        }
    }
    else {
        wxMessageBox("No row selected!", "Error", wxOK | wxICON_ERROR);
    }
}

void MyFrame::OnDelete(wxCommandEvent& event) {
    int row = grid->GetGridCursorRow();
    if (row >= 0) {
        wxString id = grid->GetCellValue(row, 0);
        if (wxMessageBox("Are you sure you want to delete this user?", "Confirm", wxYES_NO | wxICON_QUESTION) == wxYES) {
            const char* deleteQuery = "DELETE FROM Users WHERE ID = ?;";
            sqlite3_stmt* stmt;
            if (sqlite3_prepare_v2(db, deleteQuery, -1, &stmt, nullptr) == SQLITE_OK) {
                sqlite3_bind_int(stmt, 1, wxAtoi(id));
                sqlite3_step(stmt);
                sqlite3_finalize(stmt);
                LoadDataToGrid();
            }
            else {
                wxMessageBox("Failed to delete data!", "Error", wxOK | wxICON_ERROR);
            }
        }
    }
    else {
        wxMessageBox("No row selected!", "Error", wxOK | wxICON_ERROR);
    }
}

void MyFrame::OnExportXML(wxCommandEvent& event) {
    pugi::xml_document doc;
    auto root = doc.append_child("Users");

    const char* selectQuery = "SELECT * FROM Users;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, selectQuery, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            auto user = root.append_child("User");
            user.append_attribute("ID") = sqlite3_column_int(stmt, 0);
            user.append_attribute("Name") = (const char*)sqlite3_column_text(stmt, 1);
            user.append_attribute("Age") = sqlite3_column_int(stmt, 2);
        }
        sqlite3_finalize(stmt);
    }

    if (!doc.save_file("users.xml")) {
        wxMessageBox("Failed to save XML file!", "Error", wxOK | wxICON_ERROR);
    }
    else {
        wxMessageBox("Data exported to users.xml", "Success", wxOK | wxICON_INFORMATION);
    }
}
