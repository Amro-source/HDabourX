#pragma once
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <wx/wx.h>
#include <wx/notebook.h>
#include <wx/treectrl.h>
#include <wx/listctrl.h>
#include "Database.h"

class MainWindow : public wxFrame {
public:
    MainWindow();

private:
    wxNotebook* notebook;
    wxPanel* mainTab;
    wxPanel* tagsTab;
    wxTreeCtrl* tagsTree;
    wxListCtrl* notesList;
    wxButton* addNoteButton;
    wxButton* editNoteButton;
    wxButton* deleteNoteButton;
    wxButton* addTagButton;
    wxButton* addChildTagButton;

    Database db;
    void BindEvents(); // Declaration

    void SetupDatabase();
    void RefreshNotes();
    void RefreshTagsTree();
    void OnAddNote(wxCommandEvent& event);
    void OnEditNote(wxCommandEvent& event);
    void OnDeleteNote(wxCommandEvent& event);
    void OnAddTag(wxCommandEvent& event);
    void OnAddChildTag(wxCommandEvent& event);
    void OnNoteSelected(wxListEvent& event); // Declaration
    std::vector<std::pair<wxString, int>> GetAllTags(); // Declaration

    //wxTreeCtrl* tagsTree;      // Tags tree in the main tab
    wxTreeCtrl* tagsTreeTab;   // Tags tree in the tags tree tab
};

#endif