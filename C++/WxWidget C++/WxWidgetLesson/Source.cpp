#include <wx/wx.h>

// Main Application Class
class MyApp : public wxApp {
public:
    virtual bool OnInit();
};

// Main Frame (Window) Class
class MyFrame : public wxFrame {
public:
    MyFrame(const wxString& title);

private:
    void OnButtonClick(wxCommandEvent& event);

    wxTextCtrl* textBox;
};

// Macro to declare the application
wxIMPLEMENT_APP(MyApp);

// Initialize the application
bool MyApp::OnInit() {
    MyFrame* frame = new MyFrame("wxWidgets Example");
    frame->Show(true);
    return true;
}

// Define the frame (window)
MyFrame::MyFrame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(400, 300)) {
    // Create a panel inside the frame
    wxPanel* panel = new wxPanel(this, wxID_ANY);

    // Create a button
    wxButton* button = new wxButton(panel, wxID_ANY, "Click Me", wxPoint(20, 20));
    Bind(wxEVT_BUTTON, &MyFrame::OnButtonClick, this, button->GetId());

    // Create a text box
    textBox = new wxTextCtrl(panel, wxID_ANY, "", wxPoint(20, 70), wxSize(300, 30));
}

// Define button click event handler
void MyFrame::OnButtonClick(wxCommandEvent& event) {
    textBox->SetValue("Hello from wxWidgets!");
}
