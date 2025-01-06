#include <wx/wx.h>

class MyFrame : public wxFrame {
public:
    MyFrame(const wxString& title);

private:
    void OnMouseEvent(wxMouseEvent& event);

    wxPanel* m_panel;
    wxButton* m_button;
};

MyFrame::MyFrame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(800, 600);
    Centre();

    m_panel = new wxPanel(this);
    m_button = new wxButton(m_panel, wxID_ANY, "Button", wxPoint(100, 100), wxSize(100, 50));

    m_panel->Bind(wxEVT_LEFT_DOWN, &MyFrame::OnMouseEvent, this);
    m_button->Bind(wxEVT_LEFT_DOWN, &MyFrame::OnMouseEvent, this);
    CreateStatusBar();
}

void MyFrame::OnMouseEvent(wxMouseEvent& event) {
    wxPoint pos = event.GetPosition();
    wxString message = wxString::Format("Mouse clicked at (%d, %d)", pos.x, pos.y);
    SetStatusText(message);
}

class MyApp : public wxApp {
public:
    virtual bool OnInit() override;
};

bool MyApp::OnInit() {
    MyFrame* frame = new MyFrame("Mouse Events Demo");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP(MyApp);