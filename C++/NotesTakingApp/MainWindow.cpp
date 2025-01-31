#include "MainWindow.h"
#include "NoteDialog.h"
#include "TagDialog.h"
#include <functional> // Add this include for std::function

MainWindow::MainWindow() : wxFrame(nullptr, wxID_ANY, "Notes & Tags Manager") {
    // Main panel
    wxPanel* panel = new wxPanel(this);

    // Notebook (tabs)
    notebook = new wxNotebook(panel, wxID_ANY);

    // === Main Tab ===
    wxPanel* mainTab = new wxPanel(notebook);
    wxBoxSizer* mainSizer = new wxBoxSizer(wxHORIZONTAL);

    // Notes List (Left Side)
    notesList = new wxListCtrl(mainTab, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT);
    notesList->InsertColumn(0, "ID", wxLIST_FORMAT_LEFT, 50);
    notesList->InsertColumn(1, "Title", wxLIST_FORMAT_LEFT, 200);
    notesList->InsertColumn(2, "Created At", wxLIST_FORMAT_LEFT, 150);
    notesList->InsertColumn(3, "Updated At", wxLIST_FORMAT_LEFT, 150);

    // Buttons for notes
    addNoteButton = new wxButton(mainTab, wxID_ANY, "Add Note");
    editNoteButton = new wxButton(mainTab, wxID_ANY, "Edit Note");
    deleteNoteButton = new wxButton(mainTab, wxID_ANY, "Delete Note");

    wxBoxSizer* buttonSizer = new wxBoxSizer(wxHORIZONTAL);
    buttonSizer->Add(addNoteButton, 1, wxEXPAND | wxALL, 5);
    buttonSizer->Add(editNoteButton, 1, wxEXPAND | wxALL, 5);
    buttonSizer->Add(deleteNoteButton, 1, wxEXPAND | wxALL, 5);

    wxBoxSizer* notesSizer = new wxBoxSizer(wxVERTICAL);
    notesSizer->Add(notesList, 1, wxEXPAND | wxALL, 5);
    notesSizer->Add(buttonSizer, 0, wxEXPAND);
    mainSizer->Add(notesSizer, 3, wxEXPAND | wxALL, 5);

    // Tags Tree (Right Side)
    tagsTree = new wxTreeCtrl(mainTab, wxID_ANY);
    addTagButton = new wxButton(mainTab, wxID_ANY, "Add Tag");

    wxBoxSizer* tagsSizer = new wxBoxSizer(wxVERTICAL);
    tagsSizer->Add(new wxStaticText(mainTab, wxID_ANY, "Tags"), 0, wxALL, 5);
    tagsSizer->Add(tagsTree, 1, wxEXPAND | wxALL, 5);
    tagsSizer->Add(addTagButton, 0, wxEXPAND | wxALL, 5);
    mainSizer->Add(tagsSizer, 2, wxEXPAND | wxALL, 5);

    mainTab->SetSizer(mainSizer);

    // === Tags Tree Tab ===
    wxPanel* tagsTab = new wxPanel(notebook);
    wxBoxSizer* tagsTabSizer = new wxBoxSizer(wxVERTICAL);

    tagsTreeTab = new wxTreeCtrl(tagsTab, wxID_ANY);
    tagsTabSizer->Add(new wxStaticText(tagsTab, wxID_ANY, "Full Tags Tree"), 0, wxALL, 5);
    tagsTabSizer->Add(tagsTreeTab, 1, wxEXPAND | wxALL, 5);

    tagsTab->SetSizer(tagsTabSizer);

    // Add tabs to notebook
    notebook->AddPage(mainTab, "Main");
    notebook->AddPage(tagsTab, "Tags Tree");

    // Main layout
    wxBoxSizer* mainLayout = new wxBoxSizer(wxVERTICAL);
    mainLayout->Add(notebook, 1, wxEXPAND | wxALL, 5);
    panel->SetSizer(mainLayout);

    // Bind events
    BindEvents();

    // Refresh data
    RefreshNotes();
    RefreshTagsTree();
}

void MainWindow::SetupDatabase() {
    db.Execute(
        "CREATE TABLE IF NOT EXISTS Notes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "title TEXT NOT NULL, "
        "content TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    );

    db.Execute(
        "CREATE TABLE IF NOT EXISTS Tags ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT UNIQUE NOT NULL, "
        "parent_id INTEGER, "
        "FOREIGN KEY(parent_id) REFERENCES Tags(id) ON DELETE SET NULL)"
    );

    db.Execute(
        "CREATE TABLE IF NOT EXISTS NoteTags ("
        "note_id INTEGER, "
        "tag_id INTEGER, "
        "FOREIGN KEY(note_id) REFERENCES Notes(id) ON DELETE CASCADE, "
        "FOREIGN KEY(tag_id) REFERENCES Tags(id) ON DELETE CASCADE, "
        "PRIMARY KEY(note_id, tag_id))"
    );
}

void MainWindow::BindEvents() {
    // Bind the "Add Note" button to the OnAddNote handler
    addNoteButton->Bind(wxEVT_BUTTON, &MainWindow::OnAddNote, this);

    // Bind the "Edit Note" button to the OnEditNote handler
    editNoteButton->Bind(wxEVT_BUTTON, &MainWindow::OnEditNote, this);

    // Bind the "Delete Note" button to the OnDeleteNote handler
    deleteNoteButton->Bind(wxEVT_BUTTON, &MainWindow::OnDeleteNote, this);

    // Bind the "Add Tag" button to the OnAddTag handler
    addTagButton->Bind(wxEVT_BUTTON, &MainWindow::OnAddTag, this);

    // Bind the notes list selection event to the OnNoteSelected handler
    notesList->Bind(wxEVT_LIST_ITEM_SELECTED, &MainWindow::OnNoteSelected, this);
}
void MainWindow::OnNoteSelected(wxListEvent& event) {
    // Get the selected note's ID
    long selectedIndex = event.GetIndex();
    wxString noteIdStr = notesList->GetItemText(selectedIndex, 0); // Column 0 is the ID

    // Fetch the tags associated with the selected note
    auto result = db.Query(
        "SELECT Tags.name FROM Tags "
        "JOIN NoteTags ON Tags.id = NoteTags.tag_id "
        "WHERE NoteTags.note_id = ?",
        { noteIdStr }
    );

    // Clear the tags tree and display the tags for the selected note
    tagsTree->DeleteAllItems();
    wxTreeItemId root = tagsTree->AddRoot("Tags for Note");
    for (const auto& tag : result) {
        tagsTree->AppendItem(root, tag[0]);
    }
    tagsTree->Expand(root);
}

void MainWindow::RefreshNotes() {
    notesList->DeleteAllItems();
    auto notes = db.Query("SELECT id, title, created_at, updated_at FROM Notes");
    for (const auto& note : notes) {
        long index = notesList->InsertItem(0, note[0]);
        notesList->SetItem(index, 1, note[1]);
        notesList->SetItem(index, 2, note[2]);
        notesList->SetItem(index, 3, note[3]);
    }
}



void MainWindow::RefreshTagsTree() {
    // Refresh the main tab's tags tree
    tagsTree->DeleteAllItems();
    wxTreeItemId root = tagsTree->AddRoot("Tags");

    // Refresh the tags tree tab
    tagsTreeTab->DeleteAllItems();
    wxTreeItemId rootTab = tagsTreeTab->AddRoot("All Tags");

    // Function to recursively add tags to the tree
    std::function<void(wxTreeCtrl*, wxTreeItemId, int)> AddTagToTree;
    AddTagToTree = [this, &AddTagToTree](wxTreeCtrl* tree, wxTreeItemId parent, int parentId) {
        auto tags = db.Query("SELECT id, name FROM Tags WHERE parent_id = ?", { wxString::Format("%d", parentId) });
        if (tags.empty()) {
            wxLogMessage("No child tags found for parent ID: %d", parentId);
        }
        for (const auto& tag : tags) {
            wxTreeItemId item = tree->AppendItem(parent, tag[1]);
            AddTagToTree(tree, item, wxAtoi(tag[0])); // Recursively add child tags
        }
        };

    // Add top-level tags (tags with no parent)
    auto topLevelTags = db.Query("SELECT id, name FROM Tags WHERE parent_id IS -1");
    if (topLevelTags.empty()) {
        wxLogMessage("No top-level tags found.");
    }
    for (const auto& tag : topLevelTags) {
        wxTreeItemId item = tagsTree->AppendItem(root, tag[1]);
        AddTagToTree(tagsTree, item, wxAtoi(tag[0]));

        wxTreeItemId itemTab = tagsTreeTab->AppendItem(rootTab, tag[1]);
        AddTagToTree(tagsTreeTab, itemTab, wxAtoi(tag[0]));
    }

    // Expand the tree
    tagsTree->Expand(root);
    tagsTreeTab->Expand(rootTab);
}


void MainWindow::OnAddNote(wxCommandEvent& event) {
    NoteDialog dialog(this);
    if (dialog.ShowModal() == wxID_OK) {
        db.Execute("INSERT INTO Notes (title, content) VALUES (?, ?)", { dialog.GetTitle(), dialog.GetContent() });
        RefreshNotes();
    }
}

void MainWindow::OnEditNote(wxCommandEvent& event) {
    // Get the selected note from the list
    long selectedIndex = notesList->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if (selectedIndex == -1) {
        wxMessageBox("Please select a note to edit.", "Warning", wxOK | wxICON_WARNING);
        return;
    }

    // Get the note ID from the selected item
    wxString noteIdStr = notesList->GetItemText(selectedIndex, 0); // Column 0 is the ID
    long noteId;
    if (!noteIdStr.ToLong(&noteId)) {
        wxMessageBox("Invalid note ID.", "Error", wxOK | wxICON_ERROR);
        return;
    }

    // Fetch the note data from the database
    auto result = db.Query("SELECT title, content FROM Notes WHERE id = ?", { noteIdStr });
    if (result.empty() || result[0].size() < 2) {
        wxMessageBox("Failed to fetch note data.", "Error", wxOK | wxICON_ERROR);
        return;
    }

    wxString title = result[0][0];
    wxString content = result[0][1];

    // Open the edit dialog with the note data
    NoteDialog dialog(this, "Edit Note");
    dialog.SetNoteData(title, content);
    if (dialog.ShowModal() == wxID_OK) {
        // Save the updated note data
        auto noteData = dialog.GetNoteData();
        db.Execute("UPDATE Notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            { noteData.title, noteData.content, noteIdStr });
        RefreshNotes(); // Refresh the notes list
    }
}

void MainWindow::OnDeleteNote(wxCommandEvent& event) {
    // Get the selected note from the list
    long selectedIndex = notesList->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if (selectedIndex == -1) {
        wxMessageBox("Please select a note to delete.", "Warning", wxOK | wxICON_WARNING);
        return;
    }

    // Get the note ID from the selected item
    wxString noteIdStr = notesList->GetItemText(selectedIndex, 0); // Column 0 is the ID
    long noteId;
    if (!noteIdStr.ToLong(&noteId)) {
        wxMessageBox("Invalid note ID.", "Error", wxOK | wxICON_ERROR);
        return;
    }

    // Confirm deletion with the user
    wxMessageDialog confirmDialog(this, "Are you sure you want to delete this note?", "Confirm Deletion", wxYES_NO | wxICON_QUESTION);
    if (confirmDialog.ShowModal() == wxID_YES) {
        // Delete the note from the database
        db.Execute("DELETE FROM Notes WHERE id = ?", { noteIdStr });
        RefreshNotes(); // Refresh the notes list
    }
}

std::vector<std::pair<wxString, int>> MainWindow::GetAllTags() {
    std::vector<std::pair<wxString, int>> tags;
    auto result = db.Query("SELECT id, name FROM Tags");
    for (const auto& row : result) {
        int id = wxAtoi(row[0]);
        wxString name = row[1];
        tags.push_back({ name, id });
    }
    return tags;
}

void MainWindow::OnAddTag(wxCommandEvent& event) {
    // Get the selected note
    long selectedIndex = notesList->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
    if (selectedIndex == -1) {
        wxMessageBox("Please select a note to add a tag.", "Warning", wxOK | wxICON_WARNING);
        return;
    }

    // Get the note ID
    wxString noteIdStr = notesList->GetItemText(selectedIndex, 0);

    // Open the tag dialog
    TagDialog dialog(this, "Add Tag");

    // Populate the parent tags in the dialog
    auto allTags = GetAllTags(); // Retrieve all tags
    wxArrayString tagNames;
    wxArrayInt tagIds;
    for (const auto& tag : allTags) {
        tagNames.Add(tag.first); // Tag name
        tagIds.Add(tag.second);  // Tag ID
    }
    dialog.PopulateParentTags(tagNames, tagIds);

    if (dialog.ShowModal() == wxID_OK) {
        wxString tagName = dialog.GetTagName();
        int parentId = dialog.GetParentId();

        // Insert the tag into the database
        db.Execute("INSERT INTO Tags (name, parent_id) VALUES (?, ?)", { tagName, wxString::Format("%d", parentId) });

        // Link the tag to the note
        db.Execute("INSERT INTO NoteTags (note_id, tag_id) VALUES (?, (SELECT id FROM Tags WHERE name = ?))", { noteIdStr, tagName });

        // Refresh the tags tree
        RefreshTagsTree();
    }
}

void MainWindow::OnAddChildTag(wxCommandEvent& event) {
    wxTreeItemId selected = tagsTree->GetSelection();
    if (!selected.IsOk() || selected == tagsTree->GetRootItem()) {
        wxMessageBox("Please select a tag to add a child tag.", "Warning", wxOK | wxICON_WARNING);
        return;
    }

    wxString tagName = wxGetTextFromUser("Enter the name of the child tag:", "Add Child Tag");
    if (!tagName.IsEmpty()) {
        wxString parentTagName = tagsTree->GetItemText(selected);

        // Query the database for the parent tag's ID
        auto result = db.Query("SELECT id FROM Tags WHERE name = ?", { parentTagName });
        if (!result.empty() && !result[0].empty()) {
            int parentId = wxAtoi(result[0][0]); // Convert wxString to int using wxAtoi
            db.Execute("INSERT INTO Tags (name, parent_id) VALUES (?, ?)", { tagName, wxString::Format("%d", parentId) });
            RefreshTagsTree();
        }
        else {
            wxMessageBox("Parent tag not found!", "Error", wxOK | wxICON_ERROR);
        }
    }


    // Refresh the tags tree
    //RefreshTagsTree();
}