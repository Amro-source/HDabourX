#include <wx/wx.h>
#include <wx/grid.h>
#include <sqlite3.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>

// Stopwatch structure to hold individual stopwatch data
struct Stopwatch {
    std::string description;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::seconds paused_duration{ 0 };
    bool is_running = false;
    bool is_paused = false;
};

class StopwatchApp : public wxFrame {
private:
    // UI components
    wxTextCtrl* description_input;
    wxButton* start_button;
    wxButton* export_button;
    wxButton* view_data_button;
    wxScrolledWindow* scroll_panel;
    wxBoxSizer* stopwatch_sizer;

    // Stopwatch management
    std::vector<Stopwatch> stopwatches;
    std::vector<wxBoxSizer*> row_sizers;

    // Timer for updating stopwatch durations
    wxTimer* timer;

    // SQLite Database connection
    sqlite3* db;

    void InitializeDatabase() {
        int rc = sqlite3_open("stopwatch.db", &db);
        if (rc) {
            wxMessageBox("Unable to open database!", "Error", wxOK | wxICON_ERROR);
            return;
        }

        const char* create_table_query =
            "CREATE TABLE IF NOT EXISTS activities ("
            "id INTEGER PRIMARY KEY, "
            "description TEXT, "
            "start_time TEXT, "
            "end_time TEXT, "
            "duration TEXT"
            ");";

        char* errmsg = nullptr;
        rc = sqlite3_exec(db, create_table_query, nullptr, nullptr, &errmsg);
        if (rc != SQLITE_OK) {
            wxMessageBox("Error creating table!", "Error", wxOK | wxICON_ERROR);
            sqlite3_free(errmsg);
        }
    }

    std::string FormatDuration(const std::chrono::seconds& duration) {
        std::ostringstream oss;
        oss << duration.count() / 3600 << ":"
            << std::setw(2) << std::setfill('0') << (duration.count() / 60) % 60 << ":"
            << std::setw(2) << std::setfill('0') << duration.count() % 60;
        return oss.str();
    }

    void UpdateDurations(wxTimerEvent& event) {
        for (size_t i = 0; i < stopwatches.size(); ++i) {
            if (stopwatches[i].is_running && !stopwatches[i].is_paused) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - stopwatches[i].start_time
                ) + stopwatches[i].paused_duration;
                wxStaticText* duration_label = dynamic_cast<wxStaticText*>(row_sizers[i]->GetItem((int)3)->GetWindow());
                if (duration_label) {
                    duration_label->SetLabel(FormatDuration(elapsed));
                }
            }
        }
    }

    void OnStart(wxCommandEvent& event) {
        std::string description = description_input->GetValue().ToStdString();
        if (description.empty()) {
            wxMessageBox("Please enter a description!", "Error", wxOK | wxICON_ERROR);
            return;
        }

        Stopwatch sw;
        sw.description = description;
        sw.start_time = std::chrono::steady_clock::now();
        sw.is_running = true;
        stopwatches.push_back(sw);

        wxBoxSizer* row_sizer = new wxBoxSizer(wxHORIZONTAL);

        // ID
        row_sizer->Add(new wxStaticText(scroll_panel, wxID_ANY, std::to_string(stopwatches.size())), 1, wxALL | wxALIGN_CENTER_VERTICAL, 5);

        // Description
        row_sizer->Add(new wxStaticText(scroll_panel, wxID_ANY, description), 3, wxALL | wxALIGN_CENTER_VERTICAL, 5);

        // Status
        wxStaticText* status_label = new wxStaticText(scroll_panel, wxID_ANY, "Running");
        row_sizer->Add(status_label, 2, wxALL | wxALIGN_CENTER_VERTICAL, 5);

        // Duration
        wxStaticText* duration_label = new wxStaticText(scroll_panel, wxID_ANY, "0:00:00");
        row_sizer->Add(duration_label, 2, wxALL | wxALIGN_CENTER_VERTICAL, 5);

        // Buttons
        wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);
        wxButton* pause_button = new wxButton(scroll_panel, wxID_ANY, "Pause");
        wxButton* resume_button = new wxButton(scroll_panel, wxID_ANY, "Resume");
        wxButton* stop_button = new wxButton(scroll_panel, wxID_ANY, "Stop");

        resume_button->Disable(); // Initially disable Resume button
        button_sizer->Add(pause_button, 1, wxALL, 5);
        button_sizer->Add(resume_button, 1, wxALL, 5);
        button_sizer->Add(stop_button, 1, wxALL, 5);

        row_sizer->Add(button_sizer, 4, wxEXPAND | wxALL, 5);
        stopwatch_sizer->Add(row_sizer, 0, wxEXPAND | wxALL, 5);
        row_sizers.push_back(row_sizer);
        scroll_panel->FitInside();

        size_t index = stopwatches.size() - 1;

        pause_button->Bind(wxEVT_BUTTON, [=](wxCommandEvent&) {
            if (stopwatches[index].is_running && !stopwatches[index].is_paused) {
                stopwatches[index].is_paused = true;
                stopwatches[index].paused_duration += std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - stopwatches[index].start_time);
                status_label->SetLabel("Paused");
                pause_button->Disable();
                resume_button->Enable();
            }
            });

        resume_button->Bind(wxEVT_BUTTON, [=](wxCommandEvent&) {
            if (stopwatches[index].is_running && stopwatches[index].is_paused) {
                stopwatches[index].is_paused = false;
                stopwatches[index].start_time = std::chrono::steady_clock::now();
                status_label->SetLabel("Running");
                pause_button->Enable();
                resume_button->Disable();
            }
            });

        stop_button->Bind(wxEVT_BUTTON, [=](wxCommandEvent&) {
            if (stopwatches[index].is_running) {
                stopwatches[index].is_running = false;
                auto end_time = std::chrono::steady_clock::now();
                auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
                    end_time - stopwatches[index].start_time
                ) + stopwatches[index].paused_duration;

                status_label->SetLabel("Stopped");
                duration_label->SetLabel(FormatDuration(total_duration));
                pause_button->Disable();
                resume_button->Disable();
                stop_button->Disable();

                // Save to database
                std::string start_time = "START_TIMESTAMP"; // Placeholder for start timestamp
                std::string end_time_str = "END_TIMESTAMP"; // Placeholder for end timestamp
                std::string duration_str = FormatDuration(total_duration);

                const char* insert_query = "INSERT INTO activities (description, start_time, end_time, duration) VALUES (?, ?, ?, ?)";
                sqlite3_stmt* stmt;
                sqlite3_prepare_v2(db, insert_query, -1, &stmt, nullptr);
                sqlite3_bind_text(stmt, 1, stopwatches[index].description.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(stmt, 2, start_time.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(stmt, 3, end_time_str.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_text(stmt, 4, duration_str.c_str(), -1, SQLITE_STATIC);
                sqlite3_step(stmt);
                sqlite3_finalize(stmt);
            }
            });
    }

    void ExportToCSV(wxCommandEvent& event) {
        std::ofstream csv_file("stopwatch_data.csv");
        csv_file << "ID,Description,Start Time,End Time,Duration\n";

        const char* query = "SELECT * FROM activities";
        sqlite3_stmt* stmt;

        if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                int id = sqlite3_column_int(stmt, 0);
                const char* description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                const char* start_time = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                const char* end_time = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
                const char* duration = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));

                csv_file << id << ",\"" << description << "\",\"" << start_time << "\",\""
                    << end_time << "\",\"" << duration << "\"\n";
            }
            sqlite3_finalize(stmt);
            wxMessageBox("Data exported to stopwatch_data.csv", "Export Successful", wxOK | wxICON_INFORMATION);
        }
        else {
            wxMessageBox("Failed to fetch data for export", "Error", wxOK | wxICON_ERROR);
        }
    }

    void ViewData(wxCommandEvent& event) {
        wxFrame* data_frame = new wxFrame(this, wxID_ANY, "Historical Data", wxDefaultPosition, wxSize(800, 600));
        wxPanel* panel = new wxPanel(data_frame);
        wxGrid* grid = new wxGrid(panel, wxID_ANY, wxDefaultPosition, wxSize(800, 600));

        grid->CreateGrid(0, 5);
        grid->SetColLabelValue(0, "ID");
        grid->SetColLabelValue(1, "Description");
        grid->SetColLabelValue(2, "Start Time");
        grid->SetColLabelValue(3, "End Time");
        grid->SetColLabelValue(4, "Duration");

        const char* query = "SELECT * FROM activities";
        sqlite3_stmt* stmt;

        if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) == SQLITE_OK) {
            int row = 0;
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                grid->AppendRows(1);

                int id = sqlite3_column_int(stmt, 0);
                const char* description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                const char* start_time = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                const char* end_time = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
                const char* duration = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));

                grid->SetCellValue(row, 0, std::to_string(id));
                grid->SetCellValue(row, 1, description ? description : "N/A");
                grid->SetCellValue(row, 2, start_time ? start_time : "N/A");
                grid->SetCellValue(row, 3, end_time ? end_time : "N/A");
                grid->SetCellValue(row, 4, duration ? duration : "N/A");

                row++;
            }
            sqlite3_finalize(stmt);
        }
        else {
            wxMessageBox("Failed to fetch data for viewing", "Error", wxOK | wxICON_ERROR);
            return;
        }

        wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);
        vbox->Add(grid, 1, wxEXPAND | wxALL, 5);
        panel->SetSizer(vbox);
        data_frame->Show();
    }

public:
    StopwatchApp() : wxFrame(nullptr, wxID_ANY, "Stopwatch Tracker", wxDefaultPosition, wxSize(1000, 700)) {
        InitializeDatabase();

        wxPanel* panel = new wxPanel(this);
        wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

        description_input = new wxTextCtrl(panel, wxID_ANY);
        start_button = new wxButton(panel, wxID_ANY, "Start");
        export_button = new wxButton(panel, wxID_ANY, "Export Data");
        view_data_button = new wxButton(panel, wxID_ANY, "View Data");

        vbox->Add(description_input, 0, wxEXPAND | wxALL, 5);
        vbox->Add(start_button, 0, wxEXPAND | wxALL, 5);
        vbox->Add(export_button, 0, wxEXPAND | wxALL, 5);
        vbox->Add(view_data_button, 0, wxEXPAND | wxALL, 5);

        scroll_panel = new wxScrolledWindow(panel, wxID_ANY, wxDefaultPosition, wxSize(1000, 400));
        scroll_panel->SetScrollRate(5, 5);
        stopwatch_sizer = new wxBoxSizer(wxVERTICAL);
        scroll_panel->SetSizer(stopwatch_sizer);
        vbox->Add(scroll_panel, 1, wxEXPAND | wxALL, 5);

        panel->SetSizer(vbox);

        Bind(wxEVT_BUTTON, &StopwatchApp::OnStart, this, start_button->GetId());
        Bind(wxEVT_BUTTON, &StopwatchApp::ExportToCSV, this, export_button->GetId());
        Bind(wxEVT_BUTTON, &StopwatchApp::ViewData, this, view_data_button->GetId());

        timer = new wxTimer(this);
        Bind(wxEVT_TIMER, &StopwatchApp::UpdateDurations, this);
        timer->Start(1000);
    }

    ~StopwatchApp() {
        sqlite3_close(db);
    }
};

class StopwatchAppLauncher : public wxApp {
public:
    virtual bool OnInit() {
        StopwatchApp* app = new StopwatchApp();
        app->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(StopwatchAppLauncher);
