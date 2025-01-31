#pragma once
#ifndef NOTEDIALOG_H
#define NOTEDIALOG_H

#include <wx/wx.h>

class NoteDialog : public wxDialog {
public:
    NoteDialog(wxWindow* parent, const wxString& title = "Add Note");

    wxString GetTitle() const;
    wxString GetContent() const;

    // Method to get note data
    struct NoteData {
        wxString title;
        wxString content;
    };
    NoteData GetNoteData() const;


    void SetNoteData(const wxString& title, const wxString& content);

private:
    wxTextCtrl* titleCtrl;
    wxTextCtrl* contentCtrl;
};

#endif