#pragma once
#ifndef TAGDIALOG_H
#define TAGDIALOG_H

#include <wx/wx.h>

class TagDialog : public wxDialog {
public:
    TagDialog(wxWindow* parent, const wxString& title = "Add Tag");

    wxString GetTagName() const;
    int GetParentId() const;
    void PopulateParentTags(const wxArrayString& tags, const wxArrayInt& tagIds);

private:
    wxTextCtrl* tagNameCtrl;
    wxChoice* parentTagChoice;
};

#endif