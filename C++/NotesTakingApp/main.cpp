#include <wx/wx.h>
#include "MainWindow.h"

class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        MainWindow* mainWindow = new MainWindow();
        mainWindow->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(MyApp);