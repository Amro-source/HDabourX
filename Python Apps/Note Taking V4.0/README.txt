NoteApp

Overview

NoteApp is a simple note-taking application built using Python and wxPython for the GUI. It allows users to create, edit, delete, and search notes efficiently. The application also supports hierarchical tagging for better organization.

Features

Add, edit, and delete notes

Organize notes with hierarchical tags

Search notes by title, content, and tags

User-friendly wxPython GUI

SQLite database for persistent storage

Installation

Prerequisites

Ensure you have the following installed:

Python 3.x

wxPython

SQLite (bundled with Python)

Install Dependencies

pip install wxPython

Usage


Run the application using:
python NoteApp4.py

python NoteApp4.py

Database Schema

The application uses an SQLite database with the following schema:

Notes table: Stores notes with fields (id, title, content, created_at, updated_at)

Tags table: Stores tag names and hierarchical relationships

NoteTags table: Links notes to tags

Searching Notes

Users can search notes based on:

Title

Content

Associated tags

Contribution

If you'd like to contribute:

Fork the repository

Create a new branch (feature-branch)

Make your changes

Submit a pull request

License

This project is licensed under the MIT License.

