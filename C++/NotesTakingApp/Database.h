#pragma once
#ifndef DATABASE_H
#define DATABASE_H

#include <wx/wx.h>
#include <sqlite3.h>
#include <vector>

class Database {
public:
    Database(const wxString& dbPath = "notes.db");
    ~Database();

    bool Execute(const wxString& sql, const std::vector<wxString>& params = {});
    std::vector<std::vector<wxString>> Query(const wxString& sql, const std::vector<wxString>& params = {});

private:
    sqlite3* db;
    wxString dbPath;

    void Open();
    void Close();
};

#endif