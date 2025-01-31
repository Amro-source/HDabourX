#include "NoteDialog.h"

NoteDialog::NoteDialog(wxWindow* parent, const wxString& title)
    : wxDialog(parent, wxID_ANY, title, wxDefaultPosition, wxSize(400, 300)) {
    wxPanel* panel = new wxPanel(this);
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    // Title input
    sizer->Add(new wxStaticText(panel, wxID_ANY, "Title:"), 0, wxALL, 5);
    titleCtrl = new wxTextCtrl(panel, wxID_ANY);
    sizer->Add(titleCtrl, 0, wxEXPAND | wxALL, 5);

    // Content input
    sizer->Add(new wxStaticText(panel, wxID_ANY, "Content:"), 0, wxALL, 5);
    contentCtrl = new wxTextCtrl(panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE);
    sizer->Add(contentCtrl, 1, wxEXPAND | wxALL, 5);

    // Buttons
    wxBoxSizer* buttonSizer = new wxBoxSizer(wxHORIZONTAL);
    wxButton* saveButton = new wxButton(panel, wxID_OK, "Save");
    wxButton* cancelButton = new wxButton(panel, wxID_CANCEL, "Cancel");
    buttonSizer->Add(saveButton, 1, wxEXPAND | wxALL, 5);
    buttonSizer->Add(cancelButton, 1, wxEXPAND | wxALL, 5);
    sizer->Add(buttonSizer, 0, wxALIGN_CENTER);

    panel->SetSizer(sizer);
}

wxString NoteDialog::GetTitle() const {
    return titleCtrl->GetValue();
}

wxString NoteDialog::GetContent() const {
    return contentCtrl->GetValue();
}

NoteDialog::NoteData NoteDialog::GetNoteData() const {
    return { titleCtrl->GetValue(), contentCtrl->GetValue() };
}

void NoteDialog::SetNoteData(const wxString& title, const wxString& content) {
    titleCtrl->SetValue(title);
    contentCtrl->SetValue(content);
}