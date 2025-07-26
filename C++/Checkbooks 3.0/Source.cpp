//#pragma comment(lib, "C:/Users/mesho/vcpkg/installed/x64-windows/lib/sqlite3.lib")
#include "CheckbookApp.h"

class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        CheckbookApp* frame = new CheckbookApp();
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(MyApp);