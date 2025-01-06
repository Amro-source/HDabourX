#include <wx/wx.h>
#include <wx/listbox.h>

class ToDoApp : public wxApp {
public:
    virtual bool OnInit() override;
};

class ToDoFrame : public wxFrame {
public:
    ToDoFrame(const wxString& title);

private:
    wxTextCtrl* m_taskInput;      // Input field for tasks
    wxListBox* m_taskList;        // ListBox to display tasks
    wxButton* m_addButton;        // Button to add a task
    wxButton* m_deleteButton;     // Button to delete a selected task
    wxButton* m_clearButton;      // Button to clear all tasks

    void OnAddTask(wxCommandEvent& event);
    void OnDeleteTask(wxCommandEvent& event);
    void OnClearTasks(wxCommandEvent& event);

    wxDECLARE_EVENT_TABLE();
};

enum {
    ID_AddTask = 1,
    ID_DeleteTask,
    ID_ClearTasks
};

wxBEGIN_EVENT_TABLE(ToDoFrame, wxFrame)
EVT_BUTTON(ID_AddTask, ToDoFrame::OnAddTask)
EVT_BUTTON(ID_DeleteTask, ToDoFrame::OnDeleteTask)
EVT_BUTTON(ID_ClearTasks, ToDoFrame::OnClearTasks)
wxEND_EVENT_TABLE()

bool ToDoApp::OnInit() {
    ToDoFrame* frame = new ToDoFrame("To-Do Task App");
    frame->Show();
    return true;
}

ToDoFrame::ToDoFrame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(400, 300)) {
    wxPanel* panel = new wxPanel(this, wxID_ANY);

    // Create input field and buttons
    m_taskInput = new wxTextCtrl(panel, wxID_ANY, "", wxPoint(10, 10), wxSize(260, 30));
    m_addButton = new wxButton(panel, ID_AddTask, "Add Task", wxPoint(280, 10), wxSize(100, 30));
    m_taskList = new wxListBox(panel, wxID_ANY, wxPoint(10, 50), wxSize(370, 150));
    m_deleteButton = new wxButton(panel, ID_DeleteTask, "Delete Task", wxPoint(10, 210), wxSize(100, 30));
    m_clearButton = new wxButton(panel, ID_ClearTasks, "Clear Tasks", wxPoint(280, 210), wxSize(100, 30));

    CreateStatusBar();
    SetStatusText("Welcome to the To-Do Task App!");
}

void ToDoFrame::OnAddTask(wxCommandEvent& event) {
    wxString task = m_taskInput->GetValue().Trim().Trim(false);
    if (!task.IsEmpty()) {
        m_taskList->Append(task);
        m_taskInput->Clear();
        SetStatusText("Task added!");
    }
    else {
        SetStatusText("Task cannot be empty!");
    }
}

void ToDoFrame::OnDeleteTask(wxCommandEvent& event) {
    int selection = m_taskList->GetSelection();
    if (selection != wxNOT_FOUND) {
        m_taskList->Delete(selection);
        SetStatusText("Task deleted!");
    }
    else {
        SetStatusText("Please select a task to delete.");
    }
}

void ToDoFrame::OnClearTasks(wxCommandEvent& event) {
    if (m_taskList->GetCount() > 0) {
        m_taskList->Clear();
        SetStatusText("All tasks cleared!");
    }
    else {
        SetStatusText("No tasks to clear.");
    }
}

wxIMPLEMENT_APP(ToDoApp);
