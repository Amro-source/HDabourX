#include <wx/wx.h>

class mainframe : public wxFrame {
public:
    mainframe(const wxString& title);

private:
    void OnButtonClicked(wxCommandEvent& event);
    void OnSliderChanged(wxCommandEvent& event);
    void OnTextChanged(wxCommandEvent& event);

    wxDECLARE_EVENT_TABLE();
};

wxBEGIN_EVENT_TABLE(mainframe, wxFrame)
EVT_BUTTON(2, mainframe::OnButtonClicked)
EVT_SLIDER(3, mainframe::OnSliderChanged)
EVT_TEXT(4, mainframe::OnTextChanged)
wxEND_EVENT_TABLE()

mainframe::mainframe(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(800, 600);
    Centre();

    wxPanel* panel = new wxPanel(this);

    wxButton* button = new wxButton(panel, 2, "Button", wxPoint(150, 50), wxSize(100, 35));

    wxSlider* slider = new wxSlider(panel, 3, 25, 0, 100, wxPoint(0, 150), wxSize(200, -1));

    wxTextCtrl* textCtrl = new wxTextCtrl(panel, 4, "Text", wxPoint(0, 100), wxSize(200, -1));

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