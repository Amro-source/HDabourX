#include <wx/wx.h>

class MyFrame : public wxFrame {
public:
    MyFrame(const wxString& title);

private:
    void OnButtonClicked1(wxCommandEvent& event);
    void OnButtonClicked2(wxCommandEvent& event);
    void OnAnyButtonClicked(wxCommandEvent& event);
    void OnFrameClose(wxCloseEvent& event);

    wxButton* m_button1;
    wxButton* m_button2;
};

MyFrame::MyFrame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(800, 600);
    Centre();

    wxPanel* panel = new wxPanel(this);

    m_button1 = new wxButton(panel, wxID_ANY, "Button 1", wxPoint(100, 50));
    m_button2 = new wxButton(panel, wxID_ANY, "Button 2", wxPoint(250, 50));

    panel->Bind(wxEVT_CLOSE_WINDOW, &MyFrame::OnFrameClose,this);

    m_button1->Bind(wxEVT_BUTTON, &MyFrame::OnButtonClicked1, this);
    m_button2->Bind(wxEVT_BUTTON, &MyFrame::OnButtonClicked2, this);
    panel->Bind(wxEVT_BUTTON, &MyFrame::OnAnyButtonClicked, this);
}

void MyFrame::OnButtonClicked1(wxCommandEvent& event) {
    wxMessageBox("Button 1 clicked");
    // To allow event propagation, simply return without calling event.Skip()
    event.Skip();
}

void MyFrame::OnButtonClicked2(wxCommandEvent& event) {
    wxMessageBox("Button 2 clicked");
    // To allow event propagation, simply return without calling event.Skip()
    event.Skip();
    
}

void MyFrame::OnAnyButtonClicked(wxCommandEvent& event) {
    wxMessageBox("Any button clicked");
    // To stop event propagation, use event.Skip(false)
    event.Skip();
}

void MyFrame::OnFrameClose(wxCloseEvent& event) {
    wxMessageBox("Frame closed");
    // To allow the frame to close, call event.Skip()
    event.Skip();
}

class MyApp : public wxApp {
public:
    virtual bool OnInit() override;
};

bool MyApp::OnInit() {
    MyFrame* frame = new MyFrame("Event Propagation Demo");
    frame->Show();
    return true;
}

wxIMPLEMENT_APP(MyApp);