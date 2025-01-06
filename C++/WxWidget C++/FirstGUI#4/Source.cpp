#include <wx/wx.h>

class mainframe : public wxFrame {
public:
    mainframe(const wxString& title);
};

mainframe::mainframe(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(800, 600);
    Centre();

    // Create a panel to hold the controls
    wxPanel* panel = new wxPanel(this);

    // Create a button
    wxButton* button = new wxButton(panel, wxID_ANY, "Button", wxPoint(150, 50), wxSize(100, 35), wxBU_LEFT);

    // Create a checkbox
    wxCheckBox* checkbox = new wxCheckBox(panel, wxID_ANY, "Checkbox", wxPoint(550, 55), wxDefaultSize,wxCHK_3STATE | wxCHK_ALLOW_3RD_STATE_FOR_USER);

    // Create a static text control
    wxStaticText* staticText = new wxStaticText(panel, wxID_ANY, "Static Text", wxPoint(300, 100), wxSize(200, -1), wxALIGN_CENTER_HORIZONTAL);
    staticText->SetBackgroundColour(*wxLIGHT_GREY);
    // Create a text control
    wxTextCtrl* textCtrl = new wxTextCtrl(panel, wxID_ANY, "Text", wxPoint(0, 100), wxSize(200, -1), wxTE_PASSWORD);

    // Create a slider
    //wxSlider* slider = new wxSlider(panel, wxID_ANY, 25, 0, 100, wxPoint(0, 150), wxSize(200, -1), wxSL_LABEL);

    // Create a gauge
    wxGauge* gauge = new wxGauge(panel, wxID_ANY, 100, wxPoint(300, 150), wxSize(200, -1), wxGA_VERTICAL);
    gauge->SetValue(50);

    // Create an array of strings for the choice control
    wxArrayString choices;
    choices.Add("Choice 1");
    choices.Add("Choice 2");
    choices.Add("Choice 3");

    // Create a choice control
    wxChoice* choice = new wxChoice(panel, wxID_ANY, wxPoint(0, 250), wxSize(200, -1), choices, wxCB_SORT);

    // Create a spin control
    //wxSpinCtrl* spinCtrl = new wxSpinCtrl(panel, wxID_ANY, "10", wxPoint(0, 300), wxSize(80, -1), wxSP_WRAP);

    // Create a list box
    wxListBox* listBox = new wxListBox(panel, wxID_ANY, wxPoint(0, 350), wxSize(200, 100), choices, wxLB_MULTIPLE);

    // Create a radio box
    wxRadioBox* radioBox = new wxRadioBox(panel, wxID_ANY, "Radio Box", wxPoint(0, 450), wxSize(200, -1), choices, 1, wxRA_VERTICAL | wxRA_SPECIFY_ROWS);
   // radioBox->SetMajorDim(3);
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