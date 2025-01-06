#include <wx/wx.h>

// Define a new wxApp-derived class to handle the application's initialization
class MyApp : public wxApp
{
public:
    virtual bool OnInit();
};

// Define the main frame (window) of the application
class MyFrame : public wxFrame
{
public:
    MyFrame(const wxString& title);

private:
    void OnButtonClick(wxCommandEvent& event);
};

// Declare the application class
wxIMPLEMENT_APP(MyApp);

// Initialize the application
bool MyApp::OnInit()
{
    // Create the main window (frame) with the title "Hello wxWidgets"
    MyFrame* frame = new MyFrame("Hello wxWidgets");
    frame->Show(true);  // Show the frame
    return true;  // Return true to indicate the application was initialized successfully
}

// Constructor for the MyFrame class (creates the GUI)
MyFrame::MyFrame(const wxString& title)
    : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(250, 150))
{
    // Create a button with the label "Click Me"
    wxButton* button = new wxButton(this, wxID_ANY, "Click Me",
        wxPoint(50, 50), wxDefaultSize);

    // Bind the button click event to the event handler function
    Bind(wxEVT_BUTTON, &MyFrame::OnButtonClick, this);
}

// Event handler for the button click
void MyFrame::OnButtonClick(wxCommandEvent& event)
{
    wxMessageBox("You clicked the button!", "Information", wxOK | wxICON_INFORMATION);
}
