#include <wx/wx.h>

class MyFrame : public wxFrame {
public:
    MyFrame(const wxString& title);

private:
    void OnKeyEvent(wxKeyEvent& event);

    wxPanel* m_panel; // Panel to capture key events
};

MyFrame::MyFrame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(800, 600);
    Centre();

    // Create the panel
    m_panel = new wxPanel(this);
    m_panel->SetFocus(); // Ensure the panel can receive key events

    // Bind the key event handler
    m_panel->Bind(wxEVT_KEY_DOWN, &MyFrame::OnKeyEvent, this);

    CreateStatusBar();
    SetStatusText("Press keys to see key events.");
}

void MyFrame::OnKeyEvent(wxKeyEvent& event) {
    wxString message;
    wxChar keyChar = event.GetUnicodeKey();

    // Handle specific key events or log general information
    if (event.GetKeyCode() == WXK_HOME) {
        message = "Home key pressed";
    }
    else if (event.GetKeyCode() == WXK_END) {
        message = "End key pressed";
    }
    else if (keyChar >= 32 && keyChar <= 126) { // Printable ASCII range
        message.Printf("Key '%c' pressed", keyChar);
    }
    else {
        message = "Other key pressed";
    }

    SetStatusText(message);
    event.Skip(); // Allow the event to propagate further
}

class MyApp : public wxApp {
public:
    virtual bool OnInit() override;
};

bool MyApp::OnInit() {
    MyFrame* frame = new MyFrame("Keyboard Events Demo");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP(MyApp);
