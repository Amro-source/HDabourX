#include <wx/wx.h>

class MyPanel : public wxPanel {
public:
    MyPanel(wxFrame* parent);

private:
    void OnPaint(wxPaintEvent& event);

    wxDECLARE_EVENT_TABLE(); // Declare the event table
};

class MyFrame : public wxFrame {
public:
    MyFrame(const wxString& title);

private:
    MyPanel* m_panel;
};

// Define the event table
wxBEGIN_EVENT_TABLE(MyPanel, wxPanel)
EVT_PAINT(MyPanel::OnPaint)
wxEND_EVENT_TABLE()

MyPanel::MyPanel(wxFrame* parent) : wxPanel(parent) {}

void MyPanel::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);

    // Draw a rectangle
    dc.SetPen(wxPen(*wxBLACK, 2)); // Black border with 2px thickness
    dc.DrawRectangle(100, 50, 200, 100); // Rectangle coordinates (x, y, width, height)

    // Draw an ellipse
    dc.SetBrush(*wxGREEN); // Green fill
    dc.DrawEllipse(350, 100, 150, 75); // Ellipse coordinates (x, y, width, height)

    // Draw a line
    dc.SetPen(wxPen(*wxRED, 3, wxPENSTYLE_DOT)); // Red dotted line, 3px thickness
    dc.DrawLine(150, 200, 400, 200); // Line start (x1, y1) to end (x2, y2)

    // Draw text
    dc.SetTextForeground(*wxBLUE); // Blue text color
    dc.SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD)); // Font setup
    dc.DrawText("wxWidgets Drawing", 100, 250); // Draw text at position (x, y)
}

MyFrame::MyFrame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(600, 400);
    Centre();

    m_panel = new MyPanel(this); // Add the custom panel to the frame
}

class MyApp : public wxApp {
public:
    virtual bool OnInit() override;
};

bool MyApp::OnInit() {
    MyFrame* frame = new MyFrame("Drawing Demo");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP(MyApp);
