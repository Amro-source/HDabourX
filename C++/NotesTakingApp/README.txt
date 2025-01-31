Notes & Tags Manager

Notes & Tags Manager is a desktop application built with wxWidgets and SQLite for managing notes and hierarchical tags. It allows you to create, edit, and delete notes, as well as organize them using a nested tag structure. The app features a user-friendly interface with a tabbed layout for easy navigation.
Features

    Notes Management:

        Create, edit, and delete notes.

        View notes with their creation and modification timestamps.

    Hierarchical Tags:

        Create tags and nested child tags.

        Assign tags to notes for better organization.

        View a hierarchical tree of all tags in a dedicated tab.

    User Interface:

        Tabbed interface for managing notes and tags.

        Interactive tree view for tags.

        Buttons for adding child tags to existing tags.

Requirements

    C++ Compiler (e.g., GCC, Clang, or MSVC)

    wxWidgets (version 3.1 or later)

    SQLite3 (included with wxWidgets)

Usage
Notes Tab

    Add Note: Click the "Add Note" button to create a new note. Enter a title and content, then save.

    Edit Note: Select a note from the list and click "Edit Note" to modify its title or content.

    Delete Note: Select a note and click "Delete Note" to remove it.

Tags Tab

    Add Tag: Click the "Add Tag" button to create a new tag. You can optionally assign a parent tag.

    Add Child Tag: In the "Tags Tree" tab, select a tag and click "Add Child Tag" to create a nested tag under the selected one.

    View Tags: The "Tags Tree" tab displays all tags in a hierarchical structure.

Database Schema

The application uses an SQLite database (notes.db) with the following tables:

    Notes:

        id: Unique identifier for each note.

        title: Title of the note.

        content: Content of the note.

        created_at: Timestamp when the note was created.

        updated_at: Timestamp when the note was last updated.

    Tags:

        id: Unique identifier for each tag.

        name: Name of the tag.

        parent_id: ID of the parent tag (for nested tags).

    NoteTags:

        note_id: ID of the note.

        tag_id: ID of the tag assigned to the note.

Screenshots

Main Tab
Main Tab: Manage your notes and tags.

Tags Tree Tab
Tags Tree Tab: View and manage hierarchical tags.