#include <wx/wx.h>

class mainframe : public wxFrame {
public:
    mainframe(const wxString& title);

private:
    void OnButtonClicked(wxCommandEvent& event);
    void OnSliderChanged(wxCommandEvent& event);
    void OnTextChanged(wxCommandEvent& event);
};

mainframe::mainframe(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(800, 600);
    Centre();

    wxPanel* panel = new wxPanel(this);

    wxButton* button = new wxButton(panel, wxID_ANY, "Button", wxPoint(150, 50), wxSize(100, 35));
    button->Bind(wxEVT_BUTTON, &mainframe::OnButtonClicked, this);

    wxSlider* slider = new wxSlider(panel, wxID_ANY, 25, 0, 100, wxPoint(0, 150), wxSize(200, -1));
    slider->Bind(wxEVT_SLIDER, &mainframe::OnSliderChanged, this);

    wxTextCtrl* textCtrl = new wxTextCtrl(panel, wxID_ANY, "Text", wxPoint(0, 100), wxSize(200, -1));
    textCtrl->Bind(wxEVT_TEXT, &mainframe::OnTextChanged, this);

    // To remove a binding:
    // button->Unbind(wxEVT_BUTTON, &mainframe::OnButtonClicked);

    CreateStatusBar();
}

void mainframe::OnButtonClicked(wxCommandEvent& event) {
    wxLogStatus("Button clicked");
}

void mainframe::OnSliderChanged(wxCommandEvent& event) {
    wxSlider* slider = static_cast<wxSlider*>(event.GetEventObject());
    int value = slider->GetValue();
    wxString text = wxString::Format("Slider value: %d", value);
    wxLogStatus(text);
}

void mainframe::OnTextChanged(wxCommandEvent& event) {
    wxTextCtrl* textCtrl = static_cast<wxTextCtrl*>(event.GetEventObject());
    wxString text = textCtrl->GetValue();
    wxString message = wxString::Format("Text: %s", text);
    wxLogStatus(message);
}

class app : public wxApp {
public:
    virtual bool OnInit() override;
};

bool app::OnInit() {
    mainframe* frame = new mainframe("C++ GUI");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP(app);