#include <wx/wx.h>

class mainframe : public wxFrame {
public:
    mainframe(const wxString& title);
};

mainframe::mainframe(const wxString& title) : wxFrame(nullptr, wxID_ANY, title) {
    SetClientSize(800, 600);
    Centre();
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