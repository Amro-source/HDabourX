#include "Database.h"
#include <wx/msgdlg.h>

Database::Database(const wxString& dbPath) : dbPath(dbPath), db(nullptr) {
    Open();
}

Database::~Database() {
    Close();
}

void Database::Open() {
    if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
        wxMessageBox("Failed to open database!", "Error", wxOK | wxICON_ERROR);
    }
}

void Database::Close() {
    if (db) {
        sqlite3_close(db);
        db = nullptr;
    }
}

bool Database::Execute(const wxString& sql, const std::vector<wxString>& params) {
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        wxMessageBox("Failed to prepare SQL statement!", "Error", wxOK | wxICON_ERROR);
        return false;
    }

    for (size_t i = 0; i < params.size(); i++) {
        sqlite3_bind_text(stmt, i + 1, params[i].c_str(), -1, SQLITE_TRANSIENT);
    }

    bool result = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return result;
}

std::vector<std::vector<wxString>> Database::Query(const wxString& sql, const std::vector<wxString>& params) {
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        wxMessageBox("Failed to prepare SQL statement!", "Error", wxOK | wxICON_ERROR);
        return {};
    }

    for (size_t i = 0; i < params.size(); i++) {
        sqlite3_bind_text(stmt, i + 1, params[i].c_str(), -1, SQLITE_TRANSIENT);
    }

    std::vector<std::vector<wxString>> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::vector<wxString> row;
        int columnCount = sqlite3_column_count(stmt);
        for (int i = 0; i < columnCount; i++) {
            row.push_back(wxString((const char*)sqlite3_column_text(stmt, i)));
        }
        results.push_back(row);
    }

    sqlite3_finalize(stmt);
    return results;
}