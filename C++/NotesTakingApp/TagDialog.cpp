#include "TagDialog.h"

TagDialog::TagDialog(wxWindow* parent, const wxString& title)
    : wxDialog(parent, wxID_ANY, title, wxDefaultPosition, wxSize(400, 300)) {
    wxPanel* panel = new wxPanel(this);
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    // Tag name input
    sizer->Add(new wxStaticText(panel, wxID_ANY, "Tag Name:"), 0, wxALL, 5);
    tagNameCtrl = new wxTextCtrl(panel, wxID_ANY);
    sizer->Add(tagNameCtrl, 0, wxEXPAND | wxALL, 5);

    // Parent tag selection
    sizer->Add(new wxStaticText(panel, wxID_ANY, "Parent Tag (Optional):"), 0, wxALL, 5);
    parentTagChoice = new wxChoice(panel, wxID_ANY);
    sizer->Add(parentTagChoice, 0, wxEXPAND | wxALL, 5);

    // Buttons
    wxBoxSizer* buttonSizer = new wxBoxSizer(wxHORIZONTAL);
    wxButton* saveButton = new wxButton(panel, wxID_OK, "Save");
    wxButton* cancelButton = new wxButton(panel, wxID_CANCEL, "Cancel");
    buttonSizer->Add(saveButton, 1, wxEXPAND | wxALL, 5);
    buttonSizer->Add(cancelButton, 1, wxEXPAND | wxALL, 5);
    sizer->Add(buttonSizer, 0, wxALIGN_CENTER);

    panel->SetSizer(sizer);
}

wxString TagDialog::GetTagName() const {
    return tagNameCtrl->GetValue();
}

int TagDialog::GetParentId() const {
    return parentTagChoice->GetSelection() == wxNOT_FOUND ? -1 : reinterpret_cast<int>(parentTagChoice->GetClientData(parentTagChoice->GetSelection()));
}

void TagDialog::PopulateParentTags(const wxArrayString& tags, const wxArrayInt& tagIds) {
    parentTagChoice->Clear();
    parentTagChoice->Append("None", reinterpret_cast<void*>(-1));
    for (size_t i = 0; i < tags.size(); i++) {
        parentTagChoice->Append(tags[i], reinterpret_cast<void*>(tagIds[i]));
    }
    parentTagChoice->Select(0);
}